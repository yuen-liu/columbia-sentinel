"""Steering v2: calibrated latent clamping, with multi-layer and masking options.

Fixes a dosing error in v1 and adds the interventions worth testing next.

WHAT WAS WRONG IN v1
--------------------
v1 added a fixed `alpha * w_dec[f] / scale` for every feature. But the 32
tested features differ in natural activation scale by 9.2x (max_act_contact
5.1 to 47.2), so alpha=20 was 0.42x the natural range for feature 2450 and
3.90x for feature 11213. Pooling those averages "barely nudged" together with
"far off-distribution", which can wash out a real effect. v1's null is
therefore confounded, independent of whether steering works.

WHAT v2 DOES INSTEAD
--------------------
1. CALIBRATED TARGET. Each feature is driven to a target expressed in its OWN
   units: `--target-quantile` of that feature's observed activation
   distribution at contact residues (default 0.99). Every feature gets pushed
   to its own "unusually strong", not to a shared arbitrary number.

2. LATENT CLAMP + RECONSTRUCTION DIFFERENCE, rather than raw addition:
       z  = encode(x);  z' = z with z'[f] = target
       x' = x + (decode(z') - decode(z))
   This stays on the SAE's learned manifold, changes only feature f while
   leaving the other active features intact, and preserves the ~3% of variance
   the SAE cannot reconstruct (which replacing x with decode(z') would throw
   away).

3. MULTI-LAYER INJECTION (`--layers`). Single-site injection at layer 23
   leaves 7 blocks plus layernorms to suppress the perturbation; the small
   effect v1 measured is consistent with washout. Sustained injection is the
   standard remedy.
   CAVEAT, stated because it matters: the SAE is defined on layer 23. Applying
   its directions at other layers is not principled -- a residual stream is
   only partly a shared basis across layers. Injecting at layers far from 23 is
   a DIAGNOSTIC (does the effect peak at 23, or is it layer-indifferent?), not
   a fix.

4. MASKING FRACTION (`--mask-mode`, `--mask-frac`). Fully masking the binder
   maximises the leverage of the intervention -- the injected direction becomes
   the only information about it. But it also pushes the SAE out of
   distribution, since it was trained on activations from real sequences. Hence
   a sweep rather than all-or-nothing.

SUCCESS CRITERION
-----------------
Beating random directions is NOT sufficient. v1 established that a real
dictionary direction beats matched-norm noise (p=3.3e-6) purely because it
sits on the activation manifold -- the correct feature was no better than 31
unrelated ones (mean rank 15.72 vs 16.5 chance). So this script always reports
the cross-injection rank of the correct feature, which is the only measure
that distinguishes "meaningful" from "on-manifold".

Usage:
    python3 06_steering/steer_v2.py \
        --checkpoint 03_train/checkpoints/reg3a_layer23_paired/best.pt \
        --features 14364,2672,... --designs-per-feature 3 \
        --layers 23 --target-quantile 0.99 --cross-inject \
        --output-dir results/steering_v2
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
SAE_LAYER = 23
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
    ap.add_argument("--summary-csv", type=Path, default=Path("results/interface/feature_summary.csv"))
    ap.add_argument("--target-fasta", type=Path, default=Path("config/reg3a_target.fasta"))
    ap.add_argument("--features", required=True)
    ap.add_argument("--designs-per-feature", type=int, default=3)
    ap.add_argument("--skip-designs", type=int, default=0,
                    help="Skip this many of each feature's top examples before taking "
                         "--designs-per-feature. Use it to get designs held out from an "
                         "earlier run, so a subset selected there can be confirmed "
                         "without the selection inflating the result.")
    ap.add_argument("--span-before", type=int, default=2)
    ap.add_argument("--span-after", type=int, default=5)
    ap.add_argument("--layers", default=str(SAE_LAYER),
                    help="Comma-separated layers to inject at (HF hidden_states "
                         "convention). '23' reproduces v1's site; '18,...,29' tests "
                         "sustained injection; a single far layer is a diagnostic.")
    ap.add_argument("--target-quantile", type=float, default=0.99,
                    help="Drive each feature to this quantile of ITS OWN activation "
                         "distribution, instead of a shared alpha.")
    ap.add_argument("--target-multiplier", type=float, default=1.0,
                    help="Scale the calibrated target (1.0 = exactly the quantile).")
    ap.add_argument("--mask-mode", choices=["span", "binder", "frac"], default="span",
                    help="span: blank only the motif window (v1). binder: blank the "
                         "whole binder. frac: blank --mask-frac of binder positions, "
                         "always including the span.")
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--mode", choices=["clamp", "add"], default="clamp",
                    help="clamp: drive feature to its calibrated target (delta depends "
                         "on how active it already is -- CONFOUNDED for cross-injection, "
                         "see below). add: add a fixed-magnitude alpha*w_dec[f], giving "
                         "every candidate an IDENTICAL perturbation size, which is the "
                         "only version whose rank test is interpretable.\n"
                         "Measured: with clamp, the correct feature needs a 1.19x smaller "
                         "delta (it is already firing, mean act 4.23 vs 0.23), and "
                         "spearman(delta, rank) = 0.879 -- the rank tracks damage, not "
                         "meaning. Use add for cross-injection.")
    ap.add_argument("--alpha", type=float, default=20.0,
                    help="Magnitude for --mode add.")
    ap.add_argument("--cross-inject", action="store_true",
                    help="Inject every candidate feature per case and rank the correct "
                         "one. This is the real test; without it you only learn whether "
                         "the intervention beats doing nothing.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_seq = "".join(l.strip() for l in args.target_fasta.read_text().splitlines()
                         if l and not l.startswith(">")).upper()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(**ck["config"])
    sae.load_state_dict(ck["model_state_dict"])
    sae = sae.to(device).eval()
    mean, scale = ck["mean"].to(device), float(ck["scale"])

    tokenizer = EsmcTokenizer.from_pretrained(MODEL)
    model = EsmcForMaskedLM.from_pretrained(MODEL).to(device).eval()
    blocks = model.esmc.layers
    inv = {v: k for k, v in tokenizer.get_vocab().items()}
    aa_lab = [inv[i] for i in range(AA_SLICE.start, AA_SLICE.stop)]
    mask_id = model.config.mask_token_id

    manifest = pd.read_csv(args.manifest).set_index("id")
    examples = pd.read_csv(args.examples_csv)
    summary = pd.read_csv(args.summary_csv).set_index("feature")
    feats = [int(x) for x in args.features.split(",")]
    inject_layers = [int(x) for x in args.layers.split(",")]
    hook_idxs = [L - 1 for L in inject_layers]

    # Per-feature calibrated target, in that feature's own units. The quantile
    # is taken over its max-activating interface examples, which is the
    # distribution we actually have per feature.
    targets: dict[int, float] = {}
    for f in feats:
        acts = examples.loc[examples["feature"] == f, "activation"].to_numpy(dtype=float)
        if acts.size == 0:
            targets[f] = float(summary.loc[f, "max_act_contact"])
        else:
            targets[f] = float(np.quantile(acts, args.target_quantile))
        targets[f] *= args.target_multiplier
    print(f"device={device}  layers={inject_layers}  mask_mode={args.mask_mode}", flush=True)
    print(f"calibrated targets: min {min(targets.values()):.1f}  "
          f"median {np.median(list(targets.values())):.1f}  "
          f"max {max(targets.values()):.1f}  (v1 used a flat 20 for all)", flush=True)

    cases = []
    for f in feats:
        ex = (examples[examples["feature"] == f]
              .sort_values("activation", ascending=False)
              .iloc[args.skip_designs:args.skip_designs + args.designs_per_feature])
        for _, e in ex.iterrows():
            cases.append((f, e["design_id"], int(e["position"])))

    rng = np.random.RandomState(args.seed)
    rows = []
    for n, (true_feat, did, fire) in enumerate(cases, 1):
        binder = manifest.loc[did, "sequence"]
        lo, hi = max(1, fire - args.span_before), min(len(binder), fire + args.span_after)
        span = list(range(lo, hi + 1))

        if args.mask_mode == "span":
            masked_pos = list(span)
        elif args.mask_mode == "binder":
            masked_pos = list(range(1, len(binder) + 1))
        else:
            extra = [p for p in range(1, len(binder) + 1) if p not in span]
            k = max(0, int(round(args.mask_frac * len(binder))) - len(span))
            masked_pos = sorted(span + list(rng.choice(extra, size=min(k, len(extra)),
                                                       replace=False)))

        enc = tokenizer([binder + "|" + target_seq], return_tensors="pt")
        ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        masked = ids.clone()
        for q in masked_pos:
            masked[0, q] = mask_id

        native_q = [q for q in span if binder[q - 1] in aa_lab]
        native_i = [aa_lab.index(binder[q - 1]) for q in native_q]

        def run(feat_id):
            """feat_id None = no intervention. Clamp feature to its calibrated
            target and inject the reconstruction DIFFERENCE at each layer."""
            handles = []
            meas = {}

            def make_hook(is_sae_layer):
                def hook(_m, _i, out):
                    t = out[0] if isinstance(out, tuple) else out
                    if feat_id is None:
                        return out
                    t = t.clone()
                    if args.mode == "add":
                        # Identical perturbation magnitude for every candidate
                        # feature, so a cross-injection rank cannot be won just
                        # by perturbing less.
                        d = args.alpha * sae.w_dec[feat_id] / scale
                        for q in span:
                            t[0, q] = t[0, q] + d
                    else:
                        xp = (t[0] - mean) * scale
                        z = sae.encode(xp)
                        z2 = z.clone()
                        for q in span:
                            z2[q, feat_id] = targets[feat_id]
                        # Reconstruction difference: preserves what the SAE cannot
                        # explain, and perturbs only this feature's contribution.
                        rec_d = (z2 - z) @ sae.w_dec
                        t[0] = t[0] + rec_d / scale
                    if is_sae_layer:
                        meas["act"] = float(
                            sae.encode(((t[0] - mean) * scale))[span, feat_id].mean())
                    return (t,) + out[1:] if isinstance(out, tuple) else t
                return hook

            for hi_ in hook_idxs:
                handles.append(blocks[hi_].register_forward_hook(
                    make_hook(hi_ == SAE_LAYER - 1)))
            with torch.no_grad():
                out = model(input_ids=masked, attention_mask=attn)
            for h in handles:
                h.remove()
            pr = out.logits[0, :, AA_SLICE].softmax(-1).float().cpu().numpy()
            p_nat = float(np.mean([pr[q][i] for q, i in zip(native_q, native_i)]))
            pred = "".join(aa_lab[int(pr[q].argmax())] for q in span)
            return p_nat, pred, meas.get("act", float("nan"))

        base_p, base_pred, _ = run(None)
        native = "".join(binder[q - 1] for q in span)

        if args.cross_inject:
            scores = {}
            for f in feats:
                scores[f] = run(f)[0]
            order = sorted(feats, key=lambda f: -scores[f])
            rank = order.index(true_feat) + 1
            true_p, true_pred, true_act = run(true_feat)
            rows.append({
                "true_feature": true_feat, "design_id": did, "fire_pos": fire,
                "native": native, "baseline_pred": base_pred, "true_pred": true_pred,
                "baseline_p_native": round(base_p, 5),
                "true_p_native": round(true_p, 5),
                "mean_other_p_native": round(float(np.mean(
                    [scores[f] for f in feats if f != true_feat])), 5),
                "best_feature": order[0], "best_p_native": round(scores[order[0]], 5),
                "rank_of_true_feature": rank, "n_candidates": len(feats),
                "achieved_activation": round(true_act, 2),
                "calibrated_target": round(targets[true_feat], 2),
            })
        else:
            true_p, true_pred, true_act = run(true_feat)
            rows.append({
                "true_feature": true_feat, "design_id": did, "fire_pos": fire,
                "native": native, "baseline_pred": base_pred, "true_pred": true_pred,
                "baseline_p_native": round(base_p, 5),
                "true_p_native": round(true_p, 5),
                "delta_p_native": round(true_p - base_p, 5),
                "achieved_activation": round(true_act, 2),
                "calibrated_target": round(targets[true_feat], 2),
            })
        if n % 10 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)} cases", flush=True)

    df = pd.DataFrame(rows)
    tag = (f"L{'-'.join(map(str, inject_layers))}_{args.mask_mode}_"
           f"{args.mode}{args.alpha if args.mode=='add' else args.target_quantile}"
           f"{'_skip' + str(args.skip_designs) if args.skip_designs else ''}")
    df.to_csv(args.output_dir / f"steer_v2_{tag}.csv", index=False)

    from scipy import stats
    print(f"\n=== steering v2 | layers={inject_layers} mask={args.mask_mode} "
          f"q={args.target_quantile} ===", flush=True)
    print(f"cases: {len(df)}", flush=True)
    print(f"achieved activation: median {df['achieved_activation'].median():.1f} "
          f"(targets median {np.median(list(targets.values())):.1f})", flush=True)

    out_meta = {"layers": inject_layers, "mask_mode": args.mask_mode,
                "target_quantile": args.target_quantile, "n_cases": len(df),
                "cross_inject": bool(args.cross_inject)}

    if args.cross_inject:
        k = len(feats)
        exp_rank = (k + 1) / 2
        ranks = df["rank_of_true_feature"].values
        w = stats.wilcoxon(ranks - exp_rank)
        dv = (df["true_p_native"] - df["mean_other_p_native"]).values
        tv = stats.ttest_1samp(dv, 0.0)
        print(f"\nTHE TEST -- is the effect feature-SPECIFIC?", flush=True)
        print(f"  mean rank of correct feature: {ranks.mean():.2f}  "
              f"(chance {exp_rank:.1f})   wilcoxon p={w.pvalue:.3e}", flush=True)
        print(f"  ranked #1 in {int((ranks==1).sum())}/{len(df)}  "
              f"(chance {len(df)/k:.1f})", flush=True)
        print(f"  p_native vs mean of other features: {dv.mean():+.5f}  "
              f"(t p={tv.pvalue:.3e})", flush=True)
        out_meta.update(mean_rank=float(ranks.mean()), chance_rank=exp_rank,
                        wilcoxon_p=float(w.pvalue),
                        delta_vs_others=float(dv.mean()),
                        delta_vs_others_p=float(tv.pvalue))
    else:
        d = df["delta_p_native"].values
        t = stats.ttest_1samp(d, 0.0)
        print(f"\n  delta p_native vs no intervention: {d.mean():+.5f} "
              f"(t p={t.pvalue:.3e})", flush=True)
        print("  NOTE: beating 'no intervention' is not evidence of specificity. "
              "Re-run with --cross-inject for the real test.", flush=True)
        out_meta.update(delta_vs_baseline=float(d.mean()), p=float(t.pvalue))

    (args.output_dir / f"steer_v2_{tag}_meta.json").write_text(json.dumps(out_meta, indent=2))
    print(f"\nwrote {args.output_dir}/steer_v2_{tag}*", flush=True)


if __name__ == "__main__":
    main()
