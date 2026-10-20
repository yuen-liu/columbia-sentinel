"""Sweep multi-position steering: inject a feature's direction at the top-k
positions it already fires on, across k and alpha, and record whether the
design-level feature maximum actually moved.

Why this exists rather than more runs of inject_feature.py. That script
steers exactly one position per forward pass, and its results
(injection_with_controls.csv, injection_topk.csv) show the MLM argmax
essentially never changes at alpha <= ~3.6x mean_activation_when_active:
the SAE code moves (code_delta ~1.8-3.6) but the decoded residue does not.
Single-site steering at those alphas is underpowered, so there is no
sequence-level effect for the downstream structure/MPNN/energy arms to
measure.

Two separate axes, which test different claims -- see RESULTS.md's
"Interaction energy on steered mutants: run, and uninformative"
(2026-09-29), reason 1:

  alpha, at the single top-activating position
      Moves the design's feature MAXIMUM. This is the quantity the energy
      regression is actually a statement about (-13.5 kcal/mol per SD of
      `max_activation`, SD 2.08 across the 444 designs carrying 233).
  k, the number of positions steered at once
      Moves the COUNT of active sites, not necessarily the maximum --
      a design where 233 already fires at 7-21 positions will not raise its
      max just because one more site is installed. This axis tests whether
      the feature has a compositional/global sequence effect at all.

Conflating them is what made the 2026-09-29 energy run uninformative: it
perturbed the count and was read against a regression on the maximum. So
every row here carries `design_feature_max_pre` / `design_feature_max_post`
/ `design_feature_max_delta`, computed over ALL sequence positions rather
than just the interface ones. A null on this sweep is only interpretable if
those columns show the predictor moved; if max_delta ~ 0 the run says
nothing about the energy claim, exactly as before.

Controls. `--control-directions` is inherited from inject_feature.py (a
random unit direction, or another live feature's decoder row) and is the
null for "did steering do anything, or would any push do this". It is NOT
the null the downstream energy arm needs -- that one requires controls
installing the SAME residue at non-feature positions in the same design,
which is a property of the mutant set and is built by
make_control_mutants.py, not here. See RESULTS.md:1136-1145 for why an
unmatched-on-residue baseline sank both the MPNN and energy arms.

`--alpha-split` divides alpha by k so the total injected magnitude is held
constant across k. Without it, "k=4 beat k=1" is confounded by k=4 simply
having pushed 4x harder in total, which is not a compositional claim.

Usage:
    python sweep_multiposition.py \\
        --checkpoint ../checkpoints/best.pt --data-dir ../data-dir \\
        --interface-yaml ../05_interpret/interface_features.yaml \\
        --feature-stats-csv ../results/run4/feature_stats.csv \\
        --features 233 --designs-per-feature 60 \\
        --k-values 1,2,4,8,all \\
        --alpha-multipliers 1.0,2.0,4.0,8.0 \\
        --control-directions random \\
        --output sweep_multiposition_233.csv

    # matched-total-magnitude arm (compare against the run above)
    python sweep_multiposition.py ... --alpha-split \\
        --output sweep_multiposition_233_split.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: F401 -- must import before torch, see data.py's import-order note
import pandas as pd
import torch
import yaml

YAML_LOADER = yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_prepare_data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_interpret"))
from data import center_scale  # noqa: E402
from feature_analysis import load_pool, load_sae  # noqa: E402
from inject_feature import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    build_canonical_aa_vocab,
    embed_design,
    get_alphas,
    load_feature_stats,
    make_control_direction,
    resolve_features,
    select_design_feature_pairs,
    summarize_mlm_logits,
    verify_hook_alignment,
)


def make_multiposition_hook(direction_raw: torch.Tensor, alpha: float, token_indices: list[int]):
    """Adds alpha * direction_raw to several token positions in one forward
    pass. Batch size 1 only, same constraint as inject_feature.py's
    single-position hook (position indexing ignores padding across a batch).

    Deliberately a separate function rather than a generalisation of
    inject_feature.make_steering_hook: that script's outputs are already
    committed to in RESULTS.md, and widening its hook signature risks
    changing them.
    """
    index = torch.as_tensor(token_indices, dtype=torch.long, device=direction_raw.device)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output  # (1, L, D)
        hidden[0, index, :] += alpha * direction_raw
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    return hook


def feature_activations(model, tokenizer, device, sequence, layer, sae_model, mean, scale,
                        feature_id, hook=None):
    """One forward pass -> (per-position activation of `feature_id` over the
    whole sequence, MLM logits). The activation vector is over ALL residues,
    not just interface ones, because the design-level maximum is what the
    energy regression keys on and the max can sit outside the interface."""
    hidden, logits = embed_design(model, tokenizer, device, sequence, layer, hook=hook)
    hidden_proc = center_scale(hidden.to(device), mean, scale)
    with torch.no_grad():
        codes = sae_model.encode(hidden_proc)[:, feature_id]
    return codes.detach().cpu().numpy(), logits


def rank_positions(acts_pre: np.ndarray, positions: dict) -> list[tuple[int, int]]:
    """The feature's own interface positions, strongest first.

    Ranking on the live pre-activation rather than on the YAML's recorded
    values: the YAML stores a per-design `max_activation` but not a
    per-position breakdown, and the live encode is the quantity the hook
    actually perturbs."""
    ranked = sorted(positions.items(), key=lambda kv: float(acts_pre[kv[1]]), reverse=True)
    return [(int(resnum), int(position)) for resnum, position in ranked]


def process_design(
    design_id, feature_id, sequence, entry, k, k_label,
    alpha_multiplier, alpha, mean_active, aa_vocab,
    model, tokenizer, device, layer, sae_model, mean, scale, direction,
    acts_pre, logits_pre, ranked, direction_mode, reemergence_threshold, top_k,
) -> dict:
    """One (design, feature, k, alpha, direction_mode) condition.

    `acts_pre`/`logits_pre`/`ranked` are passed in because they are
    alpha- and direction-independent: one unsteered pass per design serves
    every condition, which is what makes a k x alpha x mode sweep affordable.
    """
    steered = ranked[:k]
    token_indices = [position + 1 for _, position in steered]  # +1 for CLS

    hook = make_multiposition_hook(direction, alpha, token_indices)
    acts_post, logits_post = feature_activations(
        model, tokenizer, device, sequence, layer, sae_model, mean, scale, feature_id, hook=hook
    )

    # Per-steered-position MLM read, and the multi-point mutant those
    # argmaxes imply. Applied as one simultaneous substitution rather than
    # sequentially: each argmax is conditioned on the native background, so
    # rewriting the sequence between positions would change what the later
    # ones were predicted against.
    mutant = list(sequence)
    per_position = []
    n_changed = 0
    for resnum, position in steered:
        native_aa = sequence[position]
        pre_summary = summarize_mlm_logits(logits_pre[position], aa_vocab, native_aa, top_k)
        post_summary = summarize_mlm_logits(logits_post[position], aa_vocab, native_aa, top_k)
        changed = pre_summary["argmax_aa"] != post_summary["argmax_aa"]
        n_changed += int(changed)
        mutant[position] = post_summary["argmax_aa"]
        per_position.append(
            f"{resnum}:{native_aa}>{post_summary['argmax_aa']}"
            f":{float(acts_pre[position]):.3f}>{float(acts_post[position]):.3f}"
        )
    mutant_sequence = "".join(mutant)

    # Sequence plausibility proxy. NOT pseudo-perplexity -- that needs L
    # masked forward passes and would dominate the sweep's cost. This is one
    # unsteered pass over the mutant, scoring each of its own residues under
    # the MLM head, so it is comparable BETWEEN rows of this sweep (same
    # measurement throughout) but not to a published perplexity number.
    # The real feasibility gate is pLDDT/RMSD downstream, after folding.
    if mutant_sequence == sequence:
        self_logprob = float("nan")
    else:
        _, mutant_logits = feature_activations(
            model, tokenizer, device, mutant_sequence, layer, sae_model, mean, scale, feature_id
        )
        logprobs = torch.log_softmax(mutant_logits.float(), dim=-1)
        aa_to_token = dict(aa_vocab)
        self_logprob = float(np.mean([
            logprobs[i, aa_to_token[aa]].item()
            for i, aa in enumerate(mutant_sequence) if aa in aa_to_token
        ]))

    max_pre = float(acts_pre.max())
    max_post = float(acts_post.max())
    steered_positions = [position for _, position in steered]

    return {
        "design_id": design_id, "feature_id": feature_id,
        "k_requested": k_label, "k_steered": k,
        "n_interface_positions": len(ranked),
        "direction_mode": direction_mode,
        "alpha_multiplier": alpha_multiplier, "alpha": alpha,
        "alpha_total": alpha * k,
        "mean_activation_when_active": mean_active,
        "yaml_max_activation": entry["max_activation"],
        "tier": entry["tier"],
        # The energy regression's predictor, over all positions. A null on
        # this sweep means nothing about the energy claim unless this moved.
        "design_feature_max_pre": max_pre,
        "design_feature_max_post": max_post,
        "design_feature_max_delta": max_post - max_pre,
        "design_feature_max_moved": max_post > max_pre,
        # Count of active sites -- the other axis, reported alongside so the
        # two are never conflated again.
        "n_active_pre": int((acts_pre > 0).sum()),
        "n_active_post": int((acts_post > 0).sum()),
        "steered_act_mean_pre": float(acts_pre[steered_positions].mean()),
        "steered_act_mean_post": float(acts_post[steered_positions].mean()),
        "re_emerged_frac": float(np.mean([
            acts_post[p] >= reemergence_threshold * mean_active for p in steered_positions
        ])) if np.isfinite(mean_active) else float("nan"),
        "n_aa_changed": n_changed,
        "frac_aa_changed": n_changed / k,
        "mutant_self_logprob_mean": self_logprob,
        "steered_resnums": ";".join(str(r) for r, _ in steered),
        "per_position_detail": ";".join(per_position),
        "mutant_sequence": mutant_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--interface-yaml", type=Path, required=True)
    parser.add_argument("--feature-stats-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature ids")
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--designs-per-feature", type=int, default=60,
                        help="RESULTS.md:1122 puts the minimum detectable effect at n=20 above the "
                             "effect size being probed; 60-80 is the stated target.")
    parser.add_argument("--sampling", choices=["top", "random"], default="top")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-values", type=str, default="1,2,4,8,all",
                        help="Positions steered simultaneously. 'all' steers every interface "
                             "position the feature fires on -- an upper bound expected to break, "
                             "kept in deliberately so the frontier has an endpoint.")
    parser.add_argument("--alpha-multipliers", type=str, default="1.0,2.0,4.0,8.0",
                        help="Multipliers of mean_activation_when_active. Extended well above "
                             "inject_feature.py's 0.5-1.0 range, where the argmax never moved.")
    parser.add_argument("--alpha-absolute", type=float, default=None)
    parser.add_argument("--alpha-split", action="store_true",
                        help="Divide alpha by k, holding total injected magnitude constant across "
                             "k. Run both ways: without it, a k effect is confounded by total push.")
    parser.add_argument("--control-directions", type=str, default="",
                        help="Comma-separated: random, other-feature. Paired null per condition.")
    parser.add_argument("--reemergence-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5, help="Amino acids reported per MLM readout")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", type=str, default=DEFAULT_MODEL_REVISION)
    args = parser.parse_args()

    k_values: list[tuple[str, int | None]] = []
    for token in args.k_values.split(","):
        token = token.strip()
        if not token:
            continue
        k_values.append((token, None) if token == "all" else (token, int(token)))

    from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
    from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name} on {device}...")
    model = ESMCForMaskedLM.from_pretrained(
        args.model_name, revision=args.model_revision, dtype=torch.float32
    ).to(device).eval()
    tokenizer = ESMCTokenizer()
    aa_vocab = build_canonical_aa_vocab(tokenizer)

    print("Verifying hook alignment before trusting any results from this run...")
    if not verify_hook_alignment(model, tokenizer, device, args.layer):
        raise RuntimeError("Hook-vs-hidden_states alignment check FAILED -- aborting before any injection.")

    print(f"Loading SAE checkpoint {args.checkpoint}...")
    sae_model, mean, scale, _ = load_sae(args.checkpoint, device)
    feature_stats = load_feature_stats(args.feature_stats_csv)

    print(f"Loading pool from {args.data_dir}...")
    pool_df = load_pool(args.data_dir).set_index("id")
    dup_ids = pool_df.index[pool_df.index.duplicated()].unique()
    if len(dup_ids):
        raise ValueError(f"--data-dir's manifest_combined.csv has duplicate id(s): {list(dup_ids[:3])}")

    print(f"Loading {args.interface_yaml}...")
    with open(args.interface_yaml) as f:
        yaml_data = yaml.load(f, Loader=YAML_LOADER)

    feature_ids = resolve_features(args, yaml_data["metadata"]["features_checked"])
    print(f"Sweeping {len(feature_ids)} feature(s): {feature_ids}")
    selected = select_design_feature_pairs(
        yaml_data, feature_ids, args.designs_per_feature, args.sampling, args.seed
    )

    control_modes = [m.strip() for m in args.control_directions.split(",") if m.strip()]
    unknown = [m for m in control_modes if m not in ("random", "other-feature")]
    if unknown:
        raise ValueError(f"Unknown --control-directions value(s): {unknown}")
    direction_modes = ["feature"] + control_modes
    control_rng = torch.Generator().manual_seed(args.seed + 1)

    if "dead" in feature_stats.columns:
        assert feature_stats["dead"].dtype == bool, f"'dead' column is {feature_stats['dead'].dtype}, expected bool"
        live_features = [f for f in feature_stats.index if not feature_stats.loc[f, "dead"]]
    else:
        live_features = list(feature_stats.index)

    rows = []
    n_skipped = 0
    for feature_id in feature_ids:
        alphas = get_alphas(feature_id, feature_stats, args)
        direction_raw = (sae_model.w_dec[feature_id].detach() / scale).to(device)
        mean_active = (float(feature_stats.loc[feature_id, "mean_activation_when_active"])
                       if feature_id in feature_stats.index else float("nan"))

        for n_done, entry in enumerate(selected[feature_id], start=1):
            design_id = entry["design_id"]
            if design_id not in pool_df.index:
                print(f"  SKIP {design_id}/{feature_id}: not in --data-dir pool")
                n_skipped += 1
                continue

            sequence = pool_df.loc[design_id, "sequence"]
            # One unsteered pass per design, reused by every condition below.
            acts_pre, logits_pre = feature_activations(
                model, tokenizer, device, sequence, args.layer, sae_model, mean, scale, feature_id
            )
            ranked = rank_positions(acts_pre, entry["positions"])
            if not ranked:
                print(f"  SKIP {design_id}/{feature_id}: no interface positions in YAML")
                n_skipped += 1
                continue

            for k_label, k_req in k_values:
                # k_requested and k_steered are both recorded: a k=8 row on a
                # design with only 5 interface sites is really a k=5 row, and
                # grouping on the requested label alone would blur the axis.
                k = len(ranked) if k_req is None else min(k_req, len(ranked))
                for alpha_multiplier, alpha_full in alphas:
                    alpha = alpha_full / k if args.alpha_split else alpha_full
                    for direction_mode in direction_modes:
                        direction = direction_raw if direction_mode == "feature" else make_control_direction(
                            direction_mode, direction_raw, sae_model, scale,
                            set(feature_ids), live_features, control_rng, device,
                        )
                        rows.append(process_design(
                            design_id, feature_id, sequence, entry, k, k_label,
                            alpha_multiplier, alpha, mean_active, aa_vocab,
                            model, tokenizer, device, args.layer, sae_model, mean, scale, direction,
                            acts_pre, logits_pre, ranked, direction_mode,
                            args.reemergence_threshold, args.top_k,
                        ))
            print(f"  [{n_done}/{len(selected[feature_id])}] {design_id}: "
                  f"{len(ranked)} interface position(s), {len(rows)} row(s) so far")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} row(s) to {args.output} ({n_skipped} design(s) skipped)")

    if df.empty:
        return

    # The sweep's first question is whether the energy regression's predictor
    # moved at all. Printed up front so a run that cannot speak to the energy
    # claim is obvious before anything is built on it.
    feat = df[df["direction_mode"] == "feature"]
    print("\nDid the design-level feature maximum move? (feature direction only)")
    summary = feat.groupby(["k_requested", "alpha_multiplier"]).agg(
        n=("design_id", "size"),
        max_delta_mean=("design_feature_max_delta", "mean"),
        frac_max_moved=("design_feature_max_moved", "mean"),
        frac_aa_changed=("frac_aa_changed", "mean"),
        mutant_logprob=("mutant_self_logprob_mean", "mean"),
    )
    print(summary.to_string())
    print("\nRows where the maximum did NOT move carry no information about the "
          "energy regression -- see this script's docstring and RESULTS.md:1119.")


if __name__ == "__main__":
    main()
