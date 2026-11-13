"""Load ESM-C layer-23 per-residue activations and build a train/val split
for SAE training.

Expects a directory (e.g. vilip1_layer23_per_residue/, as written by
embed_esmc.py --per-residue) containing:
    activations.npy: (total_residues, d_model) float16, memmap-written.
    index.csv: id, start, length -- activations[start:start+length] is that
        id's residues, in sequence order.
    manifest_combined.csv: id, sequence, source -- source in
        {"vilip1_full20k", "composite_hotspot", "composite_hotspot_20260728",
        "natural_binders", "binder_dataset_vilip1"}.

Rows whose source is in EVAL_ONLY_SOURCES are dropped entirely here -- those
are real UniProt sequences (13 hand-picked natural binders + 69 more pulled
from STRING functional/physical-association data, `binder_dataset_raw.csv`)
held out as a qualitative eval-only set, not part of this train/val split:
too few points to be statistically meaningful for training, and
structurally very different from the design campaigns (real protein lengths
vs. the designs' 50-80/100-150 residue binders).

vilip1_full20k and composite_hotspot are the same POI (vilip1) but
different binder-length regimes (50-80 vs. 100-150 residues) at different
scale (20000 vs. 5000 proteins, but only 68/32 by RESIDUE count since
composite_hotspot's binders are ~2x longer -- less skewed than the protein
count alone suggests). The train/val split is stratified per source (each
source split 90/10 independently, not one pooled shuffle) so both splits
keep the same source mix, and `source` is carried alongside train_x/val_x
so validation metrics can be reported per source as well as pooled.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Sources held out entirely from train/val -- real UniProt proteins used
# only for qualitative eval (see module docstring), never trained on by
# default. `natural_binders` (13 hand-picked) is NEVER trained on regardless
# of `natural_train_frac` below -- it's the more curated/trustworthy eval
# signal. `binder_dataset_vilip1` (69 STRING-derived) can be partially
# folded into training via `load_dataset`'s `natural_train_frac`.
EVAL_ONLY_SOURCES = {"natural_binders", "binder_dataset_vilip1"}
ALWAYS_EVAL_ONLY_SOURCES = {"natural_binders"}
NATURAL_PARTIAL_SOURCE = "binder_dataset_vilip1"


@dataclass
class SAEDataset:
    train_x: torch.Tensor  # (n_train, d_model) float16, RAW (uncentered, unscaled)
    val_x: torch.Tensor  # (n_val, d_model) float16, RAW
    train_source: np.ndarray  # (n_train,) str, row-aligned with train_x
    val_source: np.ndarray  # (n_val,) str, row-aligned with val_x
    mean: torch.Tensor  # (d_model,) float32, computed from train_x only
    scale: float  # computed from train_x only
    train_protein_ids: np.ndarray
    val_protein_ids: np.ndarray


def _stratified_protein_split(
    design_df: pd.DataFrame,
    val_fraction: float,
    seed: int,
    val_fraction_overrides: dict[str, float] | None = None,
) -> tuple[set, set]:
    """Returns (train_ids, val_ids) sets of protein ids, split independently
    within each `source` group so both splits keep the same source mix.

    val_fraction_overrides: per-source val fraction, falling back to
    `val_fraction` for any source not listed -- used by `load_dataset` to
    hold `binder_dataset_vilip1` out at a different rate than design
    sources when mixing natural binders into training."""
    val_fraction_overrides = val_fraction_overrides or {}
    rng = np.random.RandomState(seed)
    train_ids, val_ids = set(), set()
    for source in sorted(design_df["source"].unique()):
        frac = val_fraction_overrides.get(source, val_fraction)
        ids = design_df.loc[design_df["source"] == source, "id"].to_numpy()
        perm = rng.permutation(len(ids))
        shuffled = ids[perm]
        n_val = int(round(len(ids) * frac))
        val_ids.update(shuffled[:n_val].tolist())
        train_ids.update(shuffled[n_val:].tolist())
    return train_ids, val_ids


def _expand_to_residue_indices(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """rows: dataframe with start/length/source columns, one row per protein.
    Returns (residue_indices, residue_source) -- both length sum(rows.length),
    residue_source[i] is the source of the protein residue_indices[i] belongs to."""
    idx_parts = []
    source_parts = []
    for start, length, source in zip(rows["start"], rows["length"], rows["source"]):
        idx_parts.append(np.arange(start, start + length))
        source_parts.append(np.full(length, source, dtype=object))
    return np.concatenate(idx_parts), np.concatenate(source_parts)


def load_dataset(
    data_dir: Path, val_fraction: float = 0.1, seed: int = 0, natural_train_frac: float = 0.0
) -> SAEDataset:
    """natural_train_frac: fraction of `binder_dataset_vilip1` (69
    STRING-derived natural sequences) to fold into training, holding the
    rest out for eval alongside `natural_binders` (13 hand-picked, always
    100% held out regardless of this value). 0.0 (default) preserves the
    original behavior: both natural sources held out entirely."""
    data_dir = Path(data_dir)
    index_df = pd.read_csv(data_dir / "index.csv")
    manifest_df = pd.read_csv(data_dir / "manifest_combined.csv")[["id", "source"]]

    merged = index_df.merge(manifest_df, on="id", how="left")
    assert merged["source"].notna().all(), "some index.csv ids missing from manifest_combined.csv"

    excluded_sources = set(ALWAYS_EVAL_ONLY_SOURCES)
    if natural_train_frac <= 0:
        excluded_sources |= {NATURAL_PARTIAL_SOURCE}
    design_df = merged[~merged["source"].isin(excluded_sources)].reset_index(drop=True)
    print(
        f"{len(design_df)}/{len(merged)} proteins go into the train/val split (excluding "
        f"{len(merged) - len(design_df)} eval-only proteins from {excluded_sources}, "
        "held out for eval only)"
    )
    if natural_train_frac > 0:
        print(
            f"natural_train_frac={natural_train_frac}: {NATURAL_PARTIAL_SOURCE} is IN the "
            "split below at that train rate, NOT fully held out -- eval-only natural FVE "
            "numbers from this run aren't apples-to-apples with runs using the default 0.0"
        )

    val_fraction_overrides = {}
    if natural_train_frac > 0:
        val_fraction_overrides[NATURAL_PARTIAL_SOURCE] = 1.0 - natural_train_frac

    train_ids, val_ids = _stratified_protein_split(design_df, val_fraction, seed, val_fraction_overrides)
    train_rows = design_df[design_df["id"].isin(train_ids)]
    val_rows = design_df[design_df["id"].isin(val_ids)]
    assert set(train_rows["id"]).isdisjoint(set(val_rows["id"]))

    train_idx, train_source = _expand_to_residue_indices(train_rows)
    val_idx, val_source = _expand_to_residue_indices(val_rows)

    for name, rows, idx in [("train", train_rows, train_idx), ("val", val_rows, val_idx)]:
        counts = rows["source"].value_counts()
        print(f"{name}: {len(rows)} proteins, {len(idx)} residues -- by source: {counts.to_dict()}")

    acts = np.load(data_dir / "activations.npy")  # full load into RAM, not memmap
    train_x = torch.from_numpy(acts[train_idx].copy())
    val_x = torch.from_numpy(acts[val_idx].copy())
    del acts

    mean = train_x.float().mean(dim=0)
    centered_norms = (train_x.float() - mean).norm(dim=-1)
    scale = float(train_x.shape[1] ** 0.5 / centered_norms.mean().item())
    print(f"train mean norm={mean.norm().item():.4f}, scale={scale:.6f}")

    return SAEDataset(
        train_x=train_x,
        val_x=val_x,
        train_source=train_source,
        val_source=val_source,
        mean=mean,
        scale=scale,
        train_protein_ids=train_rows["id"].to_numpy(),
        val_protein_ids=val_rows["id"].to_numpy(),
    )


def iter_batches(x: torch.Tensor, batch_size: int, shuffle: bool = True, generator=None):
    """shuffle=True (training): random order, drops a ragged final batch.
    shuffle=False (validation): original row order preserved (so a parallel
    `source` array stays aligned), includes a final partial batch so every
    row is covered exactly once."""
    n = x.shape[0]
    if shuffle:
        perm = torch.randperm(n, generator=generator)
        n_full = (n // batch_size) * batch_size
        for i in range(0, n_full, batch_size):
            yield x[perm[i : i + batch_size]]
    else:
        for i in range(0, n, batch_size):
            yield x[i : i + batch_size]


def center_scale(x: torch.Tensor, mean: torch.Tensor, scale: float) -> torch.Tensor:
    return (x.float() - mean) * scale


def uncenter_unscale(x_proc: torch.Tensor, mean: torch.Tensor, scale: float) -> torch.Tensor:
    return x_proc / scale + mean
