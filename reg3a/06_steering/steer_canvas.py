"""Does injecting a feature make the model WRITE that feature's motif onto a
neutral canvas? A matching test with no native sequence involved.

Design
------
Canvas: a synthetic binder (poly-A by default) of a given length, co-attended
with the real Reg3A target via ESM-C's chain-break token -- the same input
form the SAE was trained on. The canvas is NOT masked: the model sees real
alanines, so its unsteered prediction is a known, flat baseline (mostly A),
and any deviation is attributable to the injection.

Injection: a feature's decoder direction at layers 23-29, optionally with a
decaying schedule (constant / linear / exponential). Rationale for decay: a
genuine activation is written once and then propagates with decay; re-stamping
it at full strength seven times is unphysical, and constant alpha=20 was
measured to produce homopolymer collapse in 23/96 cases.

Readout: for each feature we build a POSITION-SPECIFIC AMINO ACID PROFILE from
its max-activating examples (offsets -8..+8 around the firing residue,
Laplace-smoothed). After injecting feature f we score the model's predicted
distribution against every candidate feature's profile, and ask where f ranks.

Why this readout is better than the p_native one used until now
--------------------------------------------------------------
* It asks the question directly -- does the feature write ITS OWN motif --
  rather than "does it help reconstruct a sequence we already have".
* There is no native sequence, so the headroom confound (low-baseline spans
  having more room to improve, spearman -0.268 in the masked design) cannot
  arise.
* No mask tokens, so the SAE is not asked to interpret activations from an
  input form it never saw.
* Cross-injection makes the canvas's own bias harmless: all 32 candidates face
  the IDENTICAL canvas, so poly-A's helix propensity shifts every score equally
  and cancels in the ranking. (Pass --canvas-residue G for the opposite bias as
  a cross-check.)

Chance level is (n_candidates+1)/2 for mean rank, and n_cases/n_candidates for
the count of first-place finishes.

Usage:
    # pilot (NOT evidence -- used only to pick a configuration)
    python3 06_steering/steer_canvas.py --features <5 pilot ids> \
        --schedule linear --alpha 5 --output-dir results/canvas_pilot

    # pre-registered confirmation, one configuration, held-out features
    python3 06_steering/steer_canvas.py --features <27 ids> \
        --candidate-features <all 32 ids> --schedule linear --alpha 5 \
        --output-dir results/canvas_confirm
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
# Filled from the tokenizer at runtime: ESM-C vocab 4..23 is
# FREQUENCY-ordered (L,A,G,V,S,E,...), not alphabetical. Assuming
# alphabetical silently permutes every amino acid.
AA: list[str] = []


def build_profiles(examples: pd.DataFrame, manifest: pd.DataFrame,
                   feats: list[int], half: int, pseudo: float,
                   aa_order: list[str]) -> dict[int, np.ndarray]:
    """{feature: (2*half+1, 20) position-specific AA probability profile}.

    Offsets are derived from the design sequence and the firing position, not
    from the pre-rendered context string, whose window is truncated near
    sequence ends and would misalign offsets."""
    prof = {}
    for f in feats:
        counts = np.full((2 * half + 1, len(aa_order)), pseudo, dtype=float)
        ex = examples[examples["feature"] == f]
        for _, e in ex.iterrows():
            seq = manifest.loc[e["design_id"], "sequence"]
            i = int(e["position"]) - 1  # 0-based firing index
            for off in range(-half, half + 1):
                j = i + off
                if 0 <= j < len(seq) and seq[j] in aa_order:
                    counts[off + half, aa_order.index(seq[j])] += 1.0
        prof[f] = counts / counts.sum(axis=1, keepdims=True)
    return prof


def schedule_alphas(alpha: float, n_layers: int, kind: str, gamma: float) -> list[float]:
    if kind == "constant":
        return [alpha] * n_layers
    if kind == "linear":
        # alpha at the SAE layer, decaying to alpha/n_layers at the last one.
        return [alpha * (1.0 - k / n_layers) for k in range(n_layers)]
    return [alpha * (gamma ** k) for k in range(n_layers)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("results/manifest_combined.csv"))
    ap.add_argument("--examples-csv", type=Path, default=Path("results/interface/top_examples.csv"))
    ap.add_argument("--target-fasta", type=Path, default=Path("config/reg3a_target.fasta"))
    ap.add_argument("--features", required=True,
                    help="Features to TEST (each becomes the 'true' feature of a case).")
    ap.add_argument("--candidate-features", default=None,
                    help="Pool injected per case (default: --features). Keep this the "
                         "full 32 so chance level is comparable across runs.")
    ap.add_argument("--canvas-residue", default="A")
    ap.add_argument("--canvas-lengths", default="50,55,60,65,70")
    ap.add_argument("--inject-fracs", default="0.33,0.5,0.67",
                    help="Where along the canvas to inject, as a fraction of length. "
                         "Length x fraction gives independent canvases, which is what "
                         "supplies n (a synthetic canvas has no 'designs').")
    ap.add_argument("--layers", default="23,24,25,26,27,28,29")
    ap.add_argument("--schedule", choices=["constant", "linear", "exp"], default="linear")
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--inject-width", type=int, default=0,
                    help="Inject at the centre position +/- this many residues. "
                         "0 = the single firing position only.")
    ap.add_argument("--profile-half", type=int, default=8)
    ap.add_argument("--pseudocount", type=float, default=0.5)
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
    AA[:] = aa_lab  # authoritative order, straight from the tokenizer
    print(f"AA order from tokenizer: {''.join(AA)}", flush=True)

    manifest = pd.read_csv(args.manifest).set_index("id")
    examples = pd.read_csv(args.examples_csv)
    test_feats = [int(x) for x in args.features.split(",")]
    cand = ([int(x) for x in args.candidate_features.split(",")]
            if args.candidate_features else list(test_feats))
    layers = [int(x) for x in args.layers.split(",")]
    hook_idxs = [L - 1 for L in layers]
    alphas = schedule_alphas(args.alpha, len(layers), args.schedule, args.gamma)

    profiles = build_profiles(examples, manifest, sorted(set(cand) | set(test_feats)),
                              args.profile_half, args.pseudocount, AA)
    log_prof = {f: np.log(p) for f, p in profiles.items()}

    print(f"device={device}  canvas={args.canvas_residue}*L  layers={layers}", flush=True)
    print(f"schedule={args.schedule}  alphas={[round(a,2) for a in alphas]}", flush=True)
    print(f"{len(test_feats)} test features, {len(cand)} candidates "
          f"(chance rank {(len(cand)+1)/2:.1f})", flush=True)

    lengths = [int(x) for x in args.canvas_lengths.split(",")]
    fracs = [float(x) for x in args.inject_fracs.split(",")]
    canvases = [(L, fr) for L in lengths for fr in fracs]
    print(f"{len(canvases)} canvases x {len(test_feats)} features = "
          f"{len(canvases)*len(test_feats)} cases, "
          f"{len(canvases)*len(test_feats)*len(cand)} forward passes", flush=True)

    rows = []
    for n, (L, fr) in enumerate(canvases, 1):
        binder = args.canvas_residue * L
        centre = max(args.profile_half + 1,
                     min(L - args.profile_half, int(round(fr * L))))
        inj_pos = list(range(max(1, centre - args.inject_width),
                             min(L, centre + args.inject_width) + 1))
        read_pos = [centre + off for off in range(-args.profile_half, args.profile_half + 1)]

        enc = tokenizer([binder + "|" + target], return_tensors="pt")
        ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)

        def run(feat_id):
            handles = []
            for a, hi_ in zip(alphas, hook_idxs):
                d = (a * w_dec[feat_id] / scale) if feat_id is not None else None

                def hook(_m, _i, out, d=d):
                    if d is None:
                        return out
                    t = out[0] if isinstance(out, tuple) else out
                    t = t.clone()
                    for q in inj_pos:
                        t[0, q] = t[0, q] + d
                    return (t,) + out[1:] if isinstance(out, tuple) else t

                handles.append(blocks[hi_].register_forward_hook(hook))
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn)
            for h in handles:
                h.remove()
            pr = out.logits[0, :, AA_SLICE].softmax(-1).float().cpu().numpy()
            return np.stack([pr[q] for q in read_pos])  # (2*half+1, 20)

        base_dist = run(None)

        def score(dist, f):
            # Expected log-likelihood of the predicted distribution under
            # profile f: sum over read positions of <predicted, log profile>.
            return float((dist * log_prof[f]).sum())

        base_scores = {f: score(base_dist, f) for f in cand}

        for true_f in test_feats:
            dists = {f: run(f) for f in cand}
            # Score each injection against the TRUE feature's profile is the
            # wrong direction; we score the TRUE injection against every
            # candidate profile and rank. Equivalent framing, and it keeps the
            # expensive forward passes to one per candidate.
            sc = {f: score(dists[f], true_f) - base_scores[true_f] for f in cand}
            order = sorted(cand, key=lambda f: -sc[f])
            rank = order.index(true_f) + 1
            pred = "".join(AA[int(dists[true_f][k].argmax())]
                           for k in range(dists[true_f].shape[0]))
            base_pred = "".join(AA[int(base_dist[k].argmax())]
                                for k in range(base_dist.shape[0]))
            consensus = "".join(AA[int(profiles[true_f][k].argmax())]
                                for k in range(profiles[true_f].shape[0]))
            rows.append({
                "true_feature": true_f, "canvas_len": L, "inject_frac": fr,
                "centre": centre, "rank_of_true_feature": rank,
                "n_candidates": len(cand),
                "score_true": round(sc[true_f], 4),
                "score_mean_others": round(float(np.mean(
                    [sc[f] for f in cand if f != true_f])), 4),
                "best_feature": order[0], "score_best": round(sc[order[0]], 4),
                "profile_consensus": consensus,
                "baseline_pred": base_pred, "steered_pred": pred,
            })
        print(f"  canvas {n}/{len(canvases)} (L={L}, centre={centre})", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{args.canvas_residue}_{args.schedule}_a{args.alpha}_w{args.inject_width}"
    df.to_csv(args.output_dir / f"canvas_{tag}.csv", index=False)

    from scipy import stats
    k = len(cand)
    exp_rank = (k + 1) / 2
    ranks = df["rank_of_true_feature"].values
    n1 = int((ranks == 1).sum())
    wp = stats.wilcoxon(ranks - exp_rank).pvalue
    bp = stats.binomtest(n1, len(df), 1 / k, alternative="greater").pvalue
    dv = (df["score_true"] - df["score_mean_others"]).values
    tp = stats.ttest_1samp(dv, 0.0).pvalue

    print(f"\n=== canvas={args.canvas_residue} schedule={args.schedule} "
          f"alpha={args.alpha} width={args.inject_width} ===", flush=True)
    print(f"cases {len(df)}   candidates {k}   chance rank {exp_rank:.1f}", flush=True)
    print(f"  mean rank of true feature : {ranks.mean():.2f}   wilcoxon p={wp:.3e}", flush=True)
    print(f"  ranked #1                 : {n1}/{len(df)} "
          f"(chance {len(df)/k:.1f})  binomial p={bp:.3e}", flush=True)
    print(f"  score(true) - score(others): {dv.mean():+.4f}  t p={tp:.3e}", flush=True)
    print(f"\n  per-feature mean rank, best 6:", flush=True)
    g = df.groupby("true_feature")["rank_of_true_feature"].agg(["mean", "count"])
    for f, r in g.sort_values("mean").head(6).iterrows():
        print(f"    feature {f:<6} mean rank {r['mean']:.1f}  (n={int(r['count'])})", flush=True)

    (args.output_dir / f"canvas_{tag}_meta.json").write_text(json.dumps({
        "canvas_residue": args.canvas_residue, "schedule": args.schedule,
        "alpha": args.alpha, "alphas": alphas, "layers": layers,
        "inject_width": args.inject_width, "n_cases": len(df), "n_candidates": k,
        "chance_rank": exp_rank, "mean_rank": float(ranks.mean()),
        "wilcoxon_p": float(wp), "n_first": n1, "binomial_p": float(bp),
        "score_delta": float(dv.mean()), "score_delta_p": float(tp),
        "test_features": test_feats,
    }, indent=2))
    print(f"\nwrote {args.output_dir}/canvas_{tag}*", flush=True)


if __name__ == "__main__":
    main()
