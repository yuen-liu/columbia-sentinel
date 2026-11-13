"""Does the RIGHT feature steer better than a WRONG one? A matching test.

The confirmation run (steer_recover_motif.py, n=270, pre-registered) showed
that injecting a feature's decoder direction beats matched-norm RANDOM
directions on native-residue recovery (p=3.3e-6). But it did NOT show a
significant advantage over a different real feature's direction (p=0.19).

That leaves two very different explanations standing:
  (a) real dictionary directions sit on the activation manifold and therefore
      damage the representation less than random noise -- nothing to do with
      which feature it is; or
  (b) the feature actually carries the information for the motif it fires on.

This test separates them. For each (feature f, design d) where f fires at
position p, blank the span around p and inject EVERY candidate feature's
direction in turn. Then ask where f -- the feature that actually fires there --
ranks among all candidates by native-residue probability.

Under (a) the correct feature's rank is uniform: mean rank (n_features+1)/2.
Under (b) it ranks systematically better. The test is the rank itself, which
needs no matched-norm argument at all, since every injected direction is a
real unit-norm dictionary direction scaled identically.

Usage:
    python3 06_steering/cross_injection_test.py \
        --checkpoint 03_train/checkpoints/reg3a_layer23_paired/best.pt \
        --features 807,2019,... --designs-per-feature 3 --alpha 20 \
        --output-dir results/steering_cross
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_train"))
from sae_model import SparseAutoencoder  # noqa: E402

from transformers.models.esmc.modeling_esmc import EsmcForMaskedLM  # noqa: E402
from transformers.models.esmc.tokenization_esmc import EsmcTokenizer  # noqa: E402

MODEL = "biohub/ESMC-300M"
LAYER = 23
HOOK_IDX = LAYER - 1
AA_SLICE = slice(4, 24)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--examples-csv", type=Path, default=Path("results/interface/top_examples.csv"))
    ap.add_argument("--target-fasta", type=Path, default=Path("config/reg3a_target.fasta"))
    ap.add_argument("--features", required=True)
    ap.add_argument("--designs-per-feature", type=int, default=3)
    ap.add_argument("--span-before", type=int, default=2)
    ap.add_argument("--span-after", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = "".join(l.strip() for l in args.target_fasta.read_text().splitlines()
                     if l and not l.startswith(">")).upper()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(**ck["config"])
    sae.load_state_dict(ck["model_state_dict"])
    sae = sae.to(device).eval()
    mean, scale = ck["mean"].to(device), float(ck["scale"])
    w_dec = sae.w_dec.detach()

    tokenizer = EsmcTokenizer.from_pretrained(MODEL)
    model = EsmcForMaskedLM.from_pretrained(MODEL).to(device).eval()
    blocks = model.esmc.layers
    inv = {v: k for k, v in tokenizer.get_vocab().items()}
    aa_lab = [inv[i] for i in range(AA_SLICE.start, AA_SLICE.stop)]
    mask_id = model.config.mask_token_id

    manifest = pd.read_csv(args.manifest).set_index("id")
    examples = pd.read_csv(args.examples_csv)
    feats = [int(x) for x in args.features.split(",")]
    print(f"device={device}  {len(feats)} candidate directions  alpha={args.alpha}", flush=True)

    cases = []
    for f in feats:
        ex = (examples[examples["feature"] == f]
              .sort_values("activation", ascending=False)
              .head(args.designs_per_feature))
        for _, e in ex.iterrows():
            cases.append((f, e["design_id"], int(e["position"])))
    print(f"{len(cases)} cases x {len(feats)} injections = "
          f"{len(cases)*len(feats)} forward passes", flush=True)

    rows = []
    for n, (true_feat, did, fire) in enumerate(cases, 1):
        binder = manifest.loc[did, "sequence"]
        lo, hi = max(1, fire - args.span_before), min(len(binder), fire + args.span_after)
        span = list(range(lo, hi + 1))
        enc = tokenizer([binder + "|" + target], return_tensors="pt")
        ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        masked = ids.clone()
        for q in span:
            masked[0, q] = mask_id
        native_idx = [aa_lab.index(binder[q - 1]) for q in span if binder[q - 1] in aa_lab]
        native_q = [q for q in span if binder[q - 1] in aa_lab]

        def run(direction):
            def inject(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                t = t.clone()
                if direction is not None:
                    for q in span:
                        t[0, q] = t[0, q] + args.alpha * direction
                return (t,) + out[1:] if isinstance(out, tuple) else t

            h = blocks[HOOK_IDX].register_forward_hook(inject)
            with torch.no_grad():
                out = model(input_ids=masked, attention_mask=attn)
            h.remove()
            pr = out.logits[0, :, AA_SLICE].softmax(-1).float().cpu().numpy()
            return float(np.mean([pr[q][i] for q, i in zip(native_q, native_idx)]))

        baseline = run(None)
        scores = {f: run(w_dec[f] / scale) for f in feats}
        order = sorted(feats, key=lambda f: -scores[f])
        rank = order.index(true_feat) + 1
        rows.append({
            "true_feature": true_feat, "design_id": did, "fire_pos": fire,
            "span_len": len(span), "baseline_p_native": round(baseline, 5),
            "true_feature_p_native": round(scores[true_feat], 5),
            "best_feature": order[0], "best_p_native": round(scores[order[0]], 5),
            "mean_other_p_native": round(float(np.mean(
                [scores[f] for f in feats if f != true_feat])), 5),
            "rank_of_true_feature": rank, "n_candidates": len(feats),
        })
        if n % 10 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)} cases", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "cross_injection.csv", index=False)

    k = len(feats)
    exp_rank = (k + 1) / 2
    ranks = df["rank_of_true_feature"].values
    from scipy import stats
    w = stats.wilcoxon(ranks - exp_rank)
    t = stats.ttest_1samp(ranks, exp_rank)
    dvs = (df["true_feature_p_native"] - df["mean_other_p_native"]).values
    tv = stats.ttest_1samp(dvs, 0.0)

    print("\n=== does the CORRECT feature steer best? ===", flush=True)
    print(f"cases: {len(df)}   candidates per case: {k}   "
          f"expected rank under null: {exp_rank:.1f}", flush=True)
    print(f"observed mean rank of the true feature: {ranks.mean():.2f} "
          f"(median {np.median(ranks):.1f})", flush=True)
    print(f"  one-sample t vs null p = {t.pvalue:.3e}   wilcoxon p = {w.pvalue:.3e}", flush=True)
    print(f"  true feature ranked #1 in {int((ranks==1).sum())}/{len(df)} cases "
          f"(chance: {len(df)/k:.1f})", flush=True)
    print(f"  true feature in top-3 in {int((ranks<=3).sum())}/{len(df)} "
          f"(chance: {3*len(df)/k:.1f})", flush=True)
    print(f"\np_native, true feature minus mean of other features: "
          f"{dvs.mean():+.5f}  (t p = {tv.pvalue:.3e})", flush=True)

    (args.output_dir / "cross_injection_meta.json").write_text(json.dumps({
        "features": feats, "alpha": args.alpha, "n_cases": len(df),
        "expected_rank_under_null": exp_rank,
        "observed_mean_rank": float(ranks.mean()),
        "wilcoxon_p": float(w.pvalue), "ttest_p": float(t.pvalue),
        "delta_vs_other_features": float(dvs.mean()),
        "delta_vs_other_features_p": float(tv.pvalue),
    }, indent=2))
    print(f"\nwrote {args.output_dir}/cross_injection*.csv", flush=True)


if __name__ == "__main__":
    main()
