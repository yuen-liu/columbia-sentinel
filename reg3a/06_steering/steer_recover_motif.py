"""Causal test of an SAE feature: blank the region it fires in, inject its
direction at layer 23, and measure whether the model RECOVERS the motif.

Why this design, and why the obvious version fails
--------------------------------------------------
The first version of this experiment masked only one or two residues and
injected at the feature's firing position. It found no effect distinguishable
from random directions -- because the surrounding sequence was still in the
input, so the model already had the answer and the injected direction had
nothing to contribute.

Blanking the WHOLE region removes that information. The injected direction
then becomes the only source of it, and the question is sharp: does driving
this one feature make the model reconstruct the residues it otherwise gets
wrong?

Readouts, per condition:
  recovery  -- identity matches to the native region, argmax vs truth
  p_native  -- mean probability assigned to the true residues
  argmax    -- the predicted region, so motif structure is visible directly

Controls (the result is meaningless without them):
  * matched-norm random unit directions (--n-random, several seeds)
  * a different real feature's direction at the same norm
  * the no-injection masked baseline, which is what "recovery" is measured against

An alpha sweep is mandatory rather than optional: a genuine feature effect
peaks and then collapses as the representation is destroyed (observed: motif
recovered at alpha=60, degenerate poly-L at alpha=150). A random direction
has no reason to show that shape, so non-monotonicity is itself evidence.

Usage:
    python3 06_steering/steer_recover_motif.py \
        --checkpoint 03_train/checkpoints/reg3a_layer23_paired/best.pt \
        --features 14364,2672,16079,15472,9456 --designs-per-feature 3 \
        --alphas 0,20,60,150 --output-dir results/steering
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


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--examples-csv", type=Path, default=Path("results/interface/top_examples.csv"))
    ap.add_argument("--target-fasta", type=Path, default=Path("config/reg3a_target.fasta"))
    ap.add_argument("--features", required=True)
    ap.add_argument("--designs-per-feature", type=int, default=3)
    ap.add_argument("--span-before", type=int, default=2,
                    help="Residues before the firing position to blank.")
    ap.add_argument("--span-after", type=int, default=5,
                    help="Residues after the firing position to blank.")
    ap.add_argument("--alphas", default="0,20,60,150")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = "".join(l.strip() for l in args.target_fasta.read_text().splitlines()
                     if l and not l.startswith(">")).upper()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(**ck["config"])
    sae.load_state_dict(ck["model_state_dict"])
    sae = sae.to(device).eval()
    mean, scale = ck["mean"].to(device), float(ck["scale"])
    d_hidden = ck["config"]["d_hidden"]
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
    alphas = [float(a) for a in args.alphas.split(",")]
    gen = torch.Generator().manual_seed(args.seed)
    randoms = []
    for _ in range(args.n_random):
        v = torch.randn(w_dec.shape[1], generator=gen)
        randoms.append(((v / v.norm()).to(device)) / scale)

    rows = []
    for feat in feats:
        ex = (examples[examples["feature"] == feat]
              .sort_values("activation", ascending=False)
              .head(args.designs_per_feature))
        if ex.empty:
            print(f"feature {feat}: no examples, skipping", flush=True)
            continue
        other = (feat + d_hidden // 2) % d_hidden
        dirs = {"feature": w_dec[feat] / scale, f"other_feat": w_dec[other] / scale}
        for i, r in enumerate(randoms):
            dirs[f"random{i}"] = r

        for _, e in ex.iterrows():
            did, fire = e["design_id"], int(e["position"])
            binder = manifest.loc[did, "sequence"]
            lo = max(1, fire - args.span_before)
            hi = min(len(binder), fire + args.span_after)
            span = list(range(lo, hi + 1))
            native = "".join(binder[q - 1] for q in span)

            enc = tokenizer([binder + "|" + target], return_tensors="pt")
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            masked = ids.clone()
            for q in span:
                masked[0, q] = mask_id

            def run(direction, alpha):
                meas = {}

                def inject(_m, _i, out):
                    t = out[0] if isinstance(out, tuple) else out
                    t = t.clone()
                    if direction is not None and alpha:
                        for q in span:
                            t[0, q] = t[0, q] + alpha * direction
                    z = sae.encode((t[0] - mean) * scale)
                    meas["act"] = float(z[span, feat].mean())
                    return (t,) + out[1:] if isinstance(out, tuple) else t

                h = blocks[HOOK_IDX].register_forward_hook(inject)
                with torch.no_grad():
                    out = model(input_ids=masked, attention_mask=attn)
                h.remove()
                pr = out.logits[0, :, AA_SLICE].softmax(-1).float().cpu().numpy()
                pred = "".join(aa_lab[int(pr[q].argmax())] for q in span)
                p_nat = float(np.mean([pr[q][aa_lab.index(binder[q - 1])]
                                       for q in span if binder[q - 1] in aa_lab]))
                return meas["act"], pred, p_nat

            base_act, base_pred, base_pnat = run(None, 0.0)
            base_rec = sum(a == b for a, b in zip(base_pred, native))
            rows.append({"feature": feat, "design_id": did, "fire_pos": fire,
                         "native": native, "direction": "none_masked", "alpha": 0.0,
                         "pred": base_pred, "recovery": base_rec, "span_len": len(span),
                         "recovery_frac": round(base_rec / len(span), 3),
                         "p_native_mean": round(base_pnat, 4),
                         "feature_act": round(base_act, 2),
                         "delta_recovery_vs_baseline": 0,
                         "delta_p_native_vs_baseline": 0.0})

            for name, d in dirs.items():
                for alpha in alphas:
                    if alpha == 0:
                        continue
                    act, pred, pnat = run(d, alpha)
                    rec = sum(a == b for a, b in zip(pred, native))
                    rows.append({
                        "feature": feat, "design_id": did, "fire_pos": fire,
                        "native": native, "direction": name, "alpha": alpha,
                        "pred": pred, "recovery": rec, "span_len": len(span),
                        "recovery_frac": round(rec / len(span), 3),
                        "p_native_mean": round(pnat, 4),
                        "feature_act": round(act, 2),
                        "delta_recovery_vs_baseline": rec - base_rec,
                        "delta_p_native_vs_baseline": round(pnat - base_pnat, 4),
                    })
            print(f"  feature {feat} / {did} pos{fire}: native={native} "
                  f"baseline={base_pred} ({base_rec}/{len(span)})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "motif_recovery.csv", index=False)

    agg = (df[df.direction != "none_masked"]
           .groupby(["direction", "alpha"])
           .agg(mean_delta_recovery=("delta_recovery_vs_baseline", "mean"),
                mean_delta_p_native=("delta_p_native_vs_baseline", "mean"),
                mean_feature_act=("feature_act", "mean"),
                n=("delta_recovery_vs_baseline", "size"))
           .round(4).reset_index())
    agg["is_feature"] = agg["direction"] == "feature"
    agg.to_csv(args.output_dir / "motif_recovery_summary.csv", index=False)

    print("\n=== mean change vs the masked no-injection baseline ===", flush=True)
    print(agg.to_string(index=False), flush=True)

    # Headline: feature vs the pooled random controls, at each alpha.
    print("\n=== feature vs pooled random controls ===", flush=True)
    for alpha in [a for a in alphas if a]:
        f = df[(df.direction == "feature") & (df.alpha == alpha)]
        r = df[(df.direction.str.startswith("random")) & (df.alpha == alpha)]
        print(f"  alpha={alpha:>6}: feature d_recovery={f.delta_recovery_vs_baseline.mean():+.3f} "
              f"d_p_native={f.delta_p_native_vs_baseline.mean():+.4f}   |   "
              f"random d_recovery={r.delta_recovery_vs_baseline.mean():+.3f} "
              f"d_p_native={r.delta_p_native_vs_baseline.mean():+.4f}", flush=True)

    (args.output_dir / "motif_recovery_meta.json").write_text(json.dumps({
        "features": feats, "alphas": alphas, "designs_per_feature": args.designs_per_feature,
        "span_before": args.span_before, "span_after": args.span_after,
        "n_random": args.n_random, "layer": LAYER, "hook_index": HOOK_IDX,
        "scale": scale, "paired_input": True, "n_conditions": len(df),
    }, indent=2))
    print(f"\nwrote {args.output_dir}/motif_recovery*.csv", flush=True)


if __name__ == "__main__":
    main()
