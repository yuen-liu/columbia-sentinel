# Measuring latent protein representations with FVE

A protocol for asking, without labels, whether a protein language model's
hidden layer carries structured signal — and for reporting the answer so that
another team's number means the same thing as yours.

The metric is not ours. Fraction of variance explained is the standard sparse
autoencoder reconstruction measure (Gao et al., 2024). What this document
specifies is the surrounding discipline: the split, the frozen statistics, and
the parameters that have to travel with the number. Without those, FVE is a
dial you turn up by raising `k`.

---

## 1. What it measures

Train an overcomplete dictionary on per-residue activations and force each
residue to use only `k` entries. FVE is how much of the activation variance
survives that bottleneck:

```
FVE(S) = 1 − Σ_{d∈S, i} ‖u_i − û_i‖²  /  Σ_{d∈S, i} ‖u_i‖²
```

Sums run over every binder residue `i` of every design `d` in the set `S`.

A high FVE means a few sparse directions reconstruct the representation almost
perfectly — the layer's geometry is structured rather than diffuse. A low one
means the activations are not well described by a small set of directions at
that sparsity.

**The quantity is computed on centred, scaled activations**, `u = (x − μ) ⊙ s`,
not on raw hidden states. This matters for comparability: the same model and
the same dictionary give a different number if you put raw `x` in the
denominator, because the mean carries variance that no dictionary has to earn.
State which one you used.

## 2. The protocol

1. **Hold out a split before training** and never touch it. Ours is held out
   at the design level, not the residue level — residues from one design are
   highly correlated, so a residue-level split leaks.
2. **Freeze `μ` and `s` at training time** and reuse those exact statistics at
   evaluation. Recomputing them on the evaluation set quietly inflates the
   number.
3. **Report FVE with the parameters that determine it.** At minimum:

   | | ours |
   |---|---|
   | model and layer | ESM-C 300M, layer 23 of 30 |
   | `d_model` | 960 |
   | `d_hidden` | 16,384 |
   | `k` (active features per residue) | 64 |
   | split | held out by design |
   | normalisation | centred and scaled, statistics frozen at training |
   | corpus | 36,710 designs · 2,387,652 binder residues |

   An FVE quoted without `k` and `d_hidden` is not a measurement. Raising
   either raises FVE monotonically; at `k = d_model` the autoencoder is just a
   linear map and FVE goes to 1 while meaning nothing.

4. **Report dead features alongside it.** A dictionary can hit a high FVE while
   most of it is unused, which is a worse dictionary wearing a better number.
   Ours: 0 dead of 16,384.
5. **Break it down by subset.** A single pooled FVE hides specialisation.

## 3. What we found with it

**It replicates across targets.** Two different targets, two different binder
corpora, runs months apart:

| run | FVE | dead features |
|---|---|---|
| Reg3A, layer 23 | **0.9687** | 0 / 16,384 |
| VILIP-1, layer 23 | 0.9688 – 0.9730 | 18 – 31 |

Same model, same layer, same `k`, independently built pipelines. That
agreement is the reproducibility evidence — not a claim that the number is
good, but evidence that the measurement is stable under a change of target.

**It is uniform across binding modes.** Per-campaign held-out FVE on Reg3A
runs 0.94–0.99 across all ten design campaigns, so the dictionary is not
specialised to one binding mode. This breakdown is the reason to trust the
pooled figure.

**It is sensitive to a real training failure.** Dead features spiked to 2,810
at epoch 10 — exactly where `k` finishes annealing from 256 to 64 — and the
auxiliary loss drove them back to 0. The designed failure mode and its remedy
are both visible in `results/training_log_layer23_paired.csv`. A metric that
could not see that would not be worth reporting.

## 4. The caveat that matters most

**FVE measures fit to the corpus you trained on, not representation quality in
general.** Comparing our iGEM-side dictionary against Biohub's released ESM-C
autoencoder makes this unavoidable:

| dictionary | on designed binders | on natural proteins |
|---|---|---|
| ours (`run4_natural_mix`) | ~0.97 | 0.5987 |
| Biohub's released SAE | ~0.68 | 0.7276 |

Each dictionary wins on its own distribution and loses on the other. Neither
is better; they are fitted to different things. We lose our own benchmark on
natural proteins, and that comparison is off-distribution for a dictionary
trained on de novo binders — which is the point. **Report the evaluation
distribution, or the number is not interpretable.**

A corollary we state because our own data forces it: do not use FVE to select
a layer unless you have shown it discriminates on your corpus. Our layer sweep
found the criterion non-discriminating across candidate layers, so layer 23
was carried over from prior work rather than chosen by this metric. FVE
answered "does this layer carry structure" well and "which layer is best"
not at all.

## 5. Using this on your own project

The measurement needs no labels, no structures and no wet-lab data — only a
protein language model, a corpus, and a held-out split. Any team asking
whether a representation is worth building on can run it.

```bash
sentinel embed   --layer 23 ...      # per-residue activations
sentinel train   --data-dir ...      # FVE is logged per epoch
```

The held-out FVE and the dead-feature count are written to the training log
every epoch; `03_train/train.py` computes both. To reproduce our numbers from
scratch, see §11 of the README.

Adapting it is mostly a matter of substituting your own corpus. The parts that
should not change are the four in §2: hold out before training, freeze the
statistics, report `k` and `d_hidden`, and name the evaluation distribution.

## 6. Reference

Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A.,
Sutskever, I., Leike, J., & Wu, J. (2024). *Scaling and evaluating sparse
autoencoders.* arXiv:2406.04093.
