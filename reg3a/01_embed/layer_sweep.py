"""Pick the ESM-C layer to run the full pipeline on, by measuring how well
each layer's representation linearly encodes binding quality.

One forward pass per design with output_hidden_states=True yields ALL layers
at once, so sweeping 11 candidate layers costs one embedding pass over the
subsample, not eleven.

CRITERION (pre-registered, fixed before any result is seen)
-----------------------------------------------------------
Ridge regression from the mean-pooled binder embedding to each binding-quality
target, scored by cross-validated R^2 on held-out designs:
    primary:   iptm
    secondary: min_interaction_pae, binding_confidence
Winner = argmax CV R^2 on `iptm`, with ONE tie-break: if layer 23 is within
--prefer-margin (default 0.01 R^2) of the winner, layer 23 wins. Rationale --
layer 23 is the prior-work choice (it matches Biohub's own ESMC-300M SAE
exactly, keeping benchmark.py's comparison architecture-matched), so it should
only be displaced by a real margin, not by noise.

SCOPE LIMIT, stated plainly
---------------------------
PLAN.md specs a second criterion -- per-residue interface-contact
separability (AUROC for "is this residue within 5 A of the target"), which is
the criterion closest to what this project actually studies. It needs the
mmCIF contact parser that does not exist yet, so it is NOT run here. This
sweep therefore answers "which layer best linearly encodes binding quality",
not "which layer best encodes interface geometry". Re-run with the contact
criterion once the parser lands, and treat tonight's choice as provisional.

Subsample is stratified by campaign x iptm quartile so no campaign or
quality regime dominates the fit.

Usage:
    python3 01_embed/layer_sweep.py \
        --manifest results/manifest_combined.csv \
        --target-fasta config/reg3a_target.fasta \
        --layers 18,19,20,21,22,23,24,25,26,27,28 \
        --n-designs 1500 --output results/layer_sweep.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers.models.esmc.modeling_esmc import EsmcForMaskedLM
from transformers.models.esmc.tokenization_esmc import EsmcTokenizer

CHAIN_BREAK = "|"
TARGETS = ["iptm", "min_interaction_pae", "binding_confidence"]
PRIMARY = "iptm"
PRIOR_LAYER = 23


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_subsample(df: pd.DataFrame, n: int, seed: int, floor: int = 10) -> pd.DataFrame:
    """Subsample PROPORTIONAL to campaign size, stratified by iptm quartile
    within each campaign.

    Proportional, not equal-per-campaign, and the distinction matters a lot
    here. An equal-per-group scheme gives each of the 40 (campaign x quartile)
    cells the same quota, which the 7 small probe100 campaigns can always fill
    but the two 18k campaigns get capped at -- the result was 710 probe designs
    (71% of the subsample) against 296 final-campaign designs (29%), inverting
    a corpus that is 98% final-campaign designs. Since the SAE trains on that
    98%-finals corpus, the layer chosen here has to be the best layer FOR that
    corpus, so the subsample must mirror it. A `floor` keeps the small
    campaigns present (their distinct hotspots are worth having in the fit)
    without letting them drive the choice.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    sizes = df["source"].value_counts()
    total = int(sizes.sum())

    parts = []
    for source, size in sizes.items():
        quota = max(floor, int(round(n * size / total)))
        quota = min(quota, int(size))
        g = df[df["source"] == source].copy()
        # Stratify within the campaign by iptm quartile so the quota spans the
        # quality range rather than clustering wherever the data is dense.
        g["_q"] = pd.qcut(g["iptm"], 4, labels=False, duplicates="drop")
        per_q = max(1, quota // max(g["_q"].nunique(), 1))
        sub_parts = []
        for _, gq in g.groupby("_q", dropna=False):
            take = min(per_q, len(gq))
            sub_parts.append(gq.iloc[rng.choice(len(gq), size=take, replace=False)])
        parts.append(pd.concat(sub_parts).drop(columns=["_q"]))

    out = pd.concat(parts)
    if len(out) > n:
        out = out.iloc[rng.choice(len(out), size=n, replace=False)]
    return out.reset_index(drop=True)


@torch.no_grad()
def embed_all_layers(model, tokenizer, binders, target, device, layers):
    """Returns {layer: (B, d_model)} mean-pooled over binder residues only."""
    enc = tokenizer([b + CHAIN_BREAK + target for b in binders],
                    return_tensors="pt", padding=True)
    out = model(input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                output_hidden_states=True)
    pooled = {}
    for L in layers:
        h = out.hidden_states[L].float()
        pooled[L] = np.stack([h[i, 1 : 1 + len(b)].mean(0).cpu().numpy()
                              for i, b in enumerate(binders)])
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--target-fasta", type=Path, required=True)
    ap.add_argument("--layers", default="18,19,20,21,22,23,24,25,26,27,28")
    ap.add_argument("--n-designs", type=int, default=1500)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chosen-layer-json", type=Path, default=None)
    ap.add_argument("--model", default="biohub/ESMC-300M")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--prefer-margin", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    device = pick_device(args.device)
    target = "".join(l.strip() for l in args.target_fasta.read_text().splitlines()
                     if l and not l.startswith(">")).upper()

    df = pd.read_csv(args.manifest).dropna(subset=[PRIMARY])
    sub = stratified_subsample(df, args.n_designs, args.seed)
    print(f"device={device} subsample={len(sub)} layers={layers}", flush=True)
    print(sub["source"].value_counts().to_dict(), flush=True)

    tokenizer = EsmcTokenizer.from_pretrained(args.model)
    model = EsmcForMaskedLM.from_pretrained(args.model).to(device).eval()

    feats = {L: [] for L in layers}
    t0 = time.time()
    order = sub.sort_values("length", kind="stable").reset_index(drop=True)
    for i in range(0, len(order), args.batch_size):
        chunk = order.iloc[i : i + args.batch_size]
        pooled = embed_all_layers(model, tokenizer, chunk["sequence"].tolist(),
                                  target, device, layers)
        for L in layers:
            feats[L].append(pooled[L])
        if (i // args.batch_size) % 20 == 0:
            print(f"  embedded {i + len(chunk)}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    X = {L: np.concatenate(feats[L]) for L in layers}

    rows = []
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for L in layers:
        row = {"layer": L}
        for tgt in TARGETS:
            if tgt not in order.columns or order[tgt].isna().all():
                row[f"r2_{tgt}"] = np.nan
                continue
            y = order[tgt].to_numpy(dtype=float)
            ok = np.isfinite(y)
            pipe = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-2, 4, 13)),
            )
            scores = cross_val_score(pipe, X[L][ok], y[ok], cv=kf, scoring="r2")
            row[f"r2_{tgt}"] = float(scores.mean())
            row[f"r2_{tgt}_std"] = float(scores.std())
        rows.append(row)
        print(f"  layer {L}: " + "  ".join(
            f"{t}={row.get(f'r2_{t}', float('nan')):.4f}" for t in TARGETS), flush=True)

    res = pd.DataFrame(rows).sort_values("layer")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)

    col = f"r2_{PRIMARY}"
    best_row = res.loc[res[col].idxmax()]
    best_layer, best_r2 = int(best_row["layer"]), float(best_row[col])
    chosen, reason = best_layer, f"argmax CV R^2 on {PRIMARY} ({best_r2:.4f})"
    if PRIOR_LAYER in set(res["layer"]):
        prior_r2 = float(res.loc[res["layer"] == PRIOR_LAYER, col].iloc[0])
        if best_layer != PRIOR_LAYER and (best_r2 - prior_r2) <= args.prefer_margin:
            chosen = PRIOR_LAYER
            reason = (f"layer {best_layer} led by only {best_r2 - prior_r2:.4f} R^2 "
                      f"(<= margin {args.prefer_margin}); prior-work layer {PRIOR_LAYER} "
                      f"retained for Biohub benchmark comparability")

    print(f"\nCHOSEN LAYER: {chosen}\n  reason: {reason}", flush=True)
    print(res.to_string(index=False), flush=True)

    if args.chosen_layer_json:
        args.chosen_layer_json.write_text(json.dumps({
            "chosen_layer": chosen, "reason": reason, "primary_target": PRIMARY,
            "argmax_layer": best_layer, "argmax_r2": best_r2,
            "prefer_margin": args.prefer_margin, "n_designs": len(order),
            "criterion": "ridge CV R^2 from mean-pooled binder embedding",
            "scope_limit": "interface-contact AUROC criterion NOT run (parser not built); "
                           "choice is provisional",
        }, indent=2))


if __name__ == "__main__":
    main()
