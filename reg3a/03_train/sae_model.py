"""Sparse autoencoder for ESM-C layer-23 per-residue activations.

Per-token TopK architecture (Gao et al. 2024, "Scaling and evaluating sparse
autoencoders", https://arxiv.org/pdf/2406.04093) with an AuxK auxiliary loss
for dead-feature resurrection -- adapted from InterProt's implementation
(https://github.com/etowahadams/interprot/blob/main/interprot/sae_model.py),
which uses the same TopK/AuxK recipe for ESM2 activations.

Deliberately DIFFERENT from InterProt in one respect: centering here is a
FIXED, dataset-level mean subtraction computed once up front (see
data.py's `compute_center_scale`), not a per-token dynamic LayerNorm inside
the model. This follows arXiv:2605.31518's finding that activation-outlier
dimensions cause permanent feature death when a *learned* bias has to slowly
absorb the activation mean during training -- fixed centering sidesteps that
failure mode entirely. Scale normalization (a single global scalar, not
per-token) is applied alongside centering for training stability, per Gao et
al.'s convention. Both are applied OUTSIDE this module (in data.py / train.py)
so the same fixed mean/scale can be reused unchanged at inference time --
this module only ever sees already-centered-and-scaled input.

K-ANNEALING: train.py can optionally start training at a larger k than the
final target and linearly decay it down over the first slice of training
(see train.py's `--k-start`/`--k-anneal-frac`). Rationale: with a small,
fixed k from step one, which features win the per-token top-k race is
essentially random while w_enc is still near its random init, and a feature
that loses that race early and keeps losing it never gets a gradient signal
-- it's permanently dead before it had a chance to specialize into anything
useful. Larger dictionaries (this codebase moved from d_hidden=4096 towards
~10k for the vilip1 natural-binder generalization push) make this worse
because there are more features competing for the same k slots per token.
Starting with a larger k gives every feature early gradient signal, then
annealing down to the target k forces the eventual sparse code without
that early-training feature-death tax. This module only exposes the k
override on `forward()`; the schedule itself lives in train.py.

BatchTopK (Bussmann et al. 2024) was considered and deliberately rejected:
our training corpus (~1.9M residues, one narrow protein-design domain) is
too small/homogeneous for batch-relative adaptive sparsity to help, and it
would need a calibrated per-feature inference-time threshold that's noisier
to estimate on this little data. Plain per-token TopK also matches Biohub's
own general-purpose ESMC-300M SAE (fixed k=64 per residue), keeping the
benchmark comparison architecture-matched.
"""

import math
from typing import Optional

import numpy as np  # noqa: F401 -- must import before torch, see data.py's import-order note
import torch
import torch.nn as nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        k: int = 64,
        auxk: int = 256,
        dead_tokens_threshold: int = 400_000,
    ):
        """
        Args:
            d_model: dimension of the (already centered+scaled) input activations.
            d_hidden: dictionary size (number of SAE features).
            k: number of active features kept per residue (TopK).
            auxk: number of auxiliary features used to resurrect dead ones.
            dead_tokens_threshold: how many consecutive residues a feature can go
                without firing before it's considered dead and included in the
                AuxK resurrection path. Counted in individual residues (tokens),
                not steps/batches, so this is comparable across batch sizes.
        """
        super().__init__()

        self.w_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.w_dec = nn.Parameter(torch.empty(d_hidden, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.auxk = auxk
        self.dead_tokens_threshold = dead_tokens_threshold

        nn.init.kaiming_uniform_(self.w_enc, a=math.sqrt(5))
        self.w_dec.data = self.w_enc.data.T.clone()
        self.w_dec.data /= self.w_dec.data.norm(dim=1, keepdim=True)

        # For each hidden dim, how many residues have been processed since it
        # last fired. Reset to 0 whenever a batch contains a nonzero
        # activation for that feature; incremented by batch size otherwise.
        self.register_buffer("tokens_since_fired", torch.zeros(d_hidden, dtype=torch.long))

    def topk_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Per-token TopK: keep the k largest (ReLU'd) activations in the last
        dim, zero out the rest. x: (..., d_hidden)."""
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def dead_feature_mask(self) -> torch.Tensor:
        return self.tokens_since_fired > self.dead_tokens_threshold

    def forward(
        self, x: torch.Tensor, k: Optional[int] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int, torch.Tensor]:
        """x: (B, d_model) already centered+scaled. Returns (recon, auxk_recon
        or None, num_dead, latents) -- latents included so callers (e.g. the
        validation loop's L0 metric) don't need a second encode() pass.

        k: override for self.k, used by train.py's k-annealing schedule
        (larger k early in training, decaying to self.k). Validation/encode
        always use self.k (the target sparsity) regardless of this override,
        so reported metrics stay comparable across epochs and to Biohub's
        fixed k=64."""
        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topk_activation(pre_acts, k if k is not None else self.k)

        if self.training:
            # Gated on train/eval mode -- a validation forward pass must not
            # perturb this bookkeeping, or evaluating on held-out data would
            # corrupt the dead-feature signal AuxK relies on during the next
            # training step.
            fired = (latents != 0).any(dim=0)  # (d_hidden,) any residue in this batch fired it
            self.tokens_since_fired[fired] = 0
            self.tokens_since_fired[~fired] += x.shape[0]

        dead_mask = self.dead_feature_mask()
        num_dead = int(dead_mask.sum().item())

        recon = latents @ self.w_dec + self.b_dec

        if num_dead > 0:
            k_aux = min(self.auxk, num_dead)
            auxk_pre = torch.where(dead_mask[None, :], pre_acts, torch.full_like(pre_acts, -torch.inf))
            auxk_latents = self.topk_activation(auxk_pre, k_aux)
            auxk_recon = auxk_latents @ self.w_dec + self.b_dec
        else:
            auxk_recon = None

        return recon, auxk_recon, num_dead, latents

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse codes only (no reconstruction), e.g. for downstream probing."""
        pre_acts = x @ self.w_enc + self.b_enc
        return self.topk_activation(pre_acts, self.k)

    @torch.no_grad()
    def norm_weights(self) -> None:
        """Keep each decoder feature (row) unit-norm -- prevents the trivial
        trick of shrinking codes and growing decoder norm to fake low
        reconstruction loss. Call after each optimizer step.

        w_dec is (d_hidden, d_model): row i is feature i's decoder direction,
        so the norm must be taken per row (dim=1) -- collapsing dim=0 instead
        (an earlier bug here) normalizes per output activation dimension
        across all features, not per feature, and doesn't prevent the trick
        this is meant to prevent."""
        self.w_dec.data /= self.w_dec.data.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def norm_grad(self) -> None:
        """Remove the gradient component parallel to each decoder feature's
        current direction, so norm_weights's renormalization doesn't fight
        the optimizer step. Call after loss.backward(), before optimizer.step()."""
        if self.w_dec.grad is None:
            return
        dot_products = torch.sum(self.w_dec.data * self.w_dec.grad, dim=1, keepdim=True)
        self.w_dec.grad.sub_(self.w_dec.data * dot_products)


def loss_fn(
    x: torch.Tensor, recon: torch.Tensor, auxk_recon: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE reconstruction loss + AuxK auxiliary loss (coefficient 1/32, per
    Gao et al. section A.2 / InterProt's same choice)."""
    auxk_coeff = 1.0 / 32.0

    mse_loss = F.mse_loss(recon, x)
    if auxk_recon is not None:
        residual = (x - recon).detach()
        auxk_loss = auxk_coeff * F.mse_loss(auxk_recon, residual).nan_to_num(0)
    else:
        auxk_loss = torch.tensor(0.0, device=x.device)
    return mse_loss, auxk_loss
