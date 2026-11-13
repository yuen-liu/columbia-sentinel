"""Steer ESM-C by injecting an SAE feature direction at layer 23, and read the
effect out as a shift in the amino acids the model wants at those positions.

How it works
------------
ESM-C is a masked language model, so its hidden states decode back to amino
acids through its own `lm_head`. That gives a causal test of a feature's
meaning with no extra model:

  1. run the PAIRED input `binder + "|" + target`, matching exactly what the
     SAE was trained on (a binder-alone forward pass is out of distribution
     for this SAE and would invalidate the result)
  2. mask the positions being probed -- without masking, the model simply
     echoes the input residue and the shift is washed out
  3. hook layer 23 and add `alpha * w_dec[f] / scale` at those positions
  4. let layers 24-30 and `lm_head` run as usual
  5. compare the amino acid distribution at alpha=0 against alpha>0

Two readout traps, both of which produced a WRONG answer during development
---------------------------------------------------------------------------
* `output_hidden_states=True` is NOT a valid readout of a hook-modified pass.
  Verified empirically: zeroing `layers[22]`'s output leaves `hidden_states[23]`
  untouched while changing `hidden_states[24]` onward and the logits. The
  collected tuple holds pre-hook tensors. Feature activation is therefore
  measured INSIDE the hook, on the tensor actually passed downstream.
* Total-variation distance averaged over 20 amino acids x all contact
  positions is far too blunt to detect anything -- it buried the signal
  entirely. Use the feature's own motif to make a sharp, falsifiable
  prediction about specific amino acids at specific offsets.

Two details that silently corrupt the result if missed
-----------------------------------------------------
* SPACE. The SAE is trained on centered-and-scaled activations,
  `x_proc = (x - mean) * scale`, and `w_dec` rows are unit-norm in THAT
  space. Injecting into the raw hidden state therefore needs `/ scale`
  (scale ~= 0.09, so omitting it misstates alpha by ~11x).
* LAYER INDEX. HuggingFace's `hidden_states[k]` is the output of `layers[k-1]`
  (`hidden_states[0]` is the embedding). So layer 23 is `layers[22]`. This is
  asserted at runtime rather than assumed.

Controls, which are the difference between a result and a light show
--------------------------------------------------------------------
Any sufficiently large perturbation changes the output. `--controls` adds, at
matched norm:
  * `random`  -- a random unit direction (several seeds)
  * `other`   -- a different feature's decoder direction
If the chosen feature's effect is not distinguishable from these, the demo
shows nothing. The alpha sweep is the other half: a monotone dose-response is
much harder to dismiss than one hand-picked alpha.

Usage:
    python3 06_steering/steer_esmc.py \
        --checkpoint 03_train/checkpoints/reg3a_layer23_paired/best.pt \
        --feature 14364 --design-id pres_... \
        --manifest results/manifest_combined.csv \
        --interface-cache results/interface/iface_cache_*.pkl \
        --alphas 0,5,10,20,40 --controls \
        --output-dir results/steering
"""

import argparse
import json
import pickle
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
CHAIN_BREAK = "|"
LAYER = 23
AA_SLICE = slice(4, 24)  # the 20 amino acids in ESM-C's vocab, per binder_design.py
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sae(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(**ck["config"])
    sae.load_state_dict(ck["model_state_dict"])
    return (sae.to(device).eval(), ck["mean"].to(device), float(ck["scale"]),
            ck["config"]["d_hidden"])


def aa_labels(tokenizer) -> list[str]:
    """Vocab ids 4..23 in order, mapped to single letters."""
    inv = {v: k for k, v in tokenizer.get_vocab().items()}
    return [inv.get(i, f"?{i}") for i in range(AA_SLICE.start, AA_SLICE.stop)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--target-fasta", type=Path, default=Path("config/reg3a_target.fasta"))
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--design-id", default=None,
                    help="Which design to steer. Default: the design holding this "
                         "feature's single hardest-firing interface residue.")
    ap.add_argument("--examples-csv", type=Path, default=Path("results/interface/top_examples.csv"))
    ap.add_argument("--interface-cache", type=Path, default=None,
                    help="Geometry pickle from interface_activations.py; used to "
                         "restrict injection to contact residues.")
    ap.add_argument("--positions", default=None,
                    help="Comma-separated 1-based binder positions to mask+steer. "
                         "Default: the contact residues from --interface-cache, or "
                         "this feature's own firing position.")
    ap.add_argument("--inject-positions", default=None,
                    help="1-based binder positions to ADD the direction at "
                         "(default: this feature's own hardest-firing position).")
    ap.add_argument("--probe-positions", default=None,
                    help="1-based binder positions to MASK and read predictions "
                         "at (default: inject_position + 1 and + 2, which is where "
                         "a motif-following residue would show up).")
    ap.add_argument("--alphas", default="0,5,20,60,150")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--other-feature", type=int, default=None)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = "".join(l.strip() for l in args.target_fasta.read_text().splitlines()
                     if l and not l.startswith(">")).upper()

    sae, mean, scale, d_hidden = load_sae(args.checkpoint, device)
    manifest = pd.read_csv(args.manifest).set_index("id")

    # ---- pick the design and the positions to probe ----
    design_id = args.design_id
    fire_pos = None
    if design_id is None and args.examples_csv.is_file():
        ex = pd.read_csv(args.examples_csv)
        ex = ex[ex["feature"] == args.feature].sort_values("activation", ascending=False)
        if ex.empty:
            raise SystemExit(f"feature {args.feature} has no examples in {args.examples_csv}")
        design_id, fire_pos = ex.iloc[0]["design_id"], int(ex.iloc[0]["position"])
    if design_id not in manifest.index:
        raise SystemExit(f"design {design_id} not in manifest")
    binder = manifest.loc[design_id, "sequence"]

    if args.positions:
        positions = [int(p) for p in args.positions.split(",")]
    elif args.interface_cache:
        cache = sorted(Path().glob(str(args.interface_cache)))
        path = cache[0] if cache else args.interface_cache
        with open(path, "rb") as fh:
            results, _ = pickle.load(fh)
        if design_id not in results:
            raise SystemExit(f"{design_id} not in interface cache")
        positions = [i + 1 for i, t in enumerate(results[design_id]["tiers"]) if t == "contact"]
    else:
        positions = [fire_pos]
    positions = [p for p in positions if 1 <= p <= len(binder)]
    if not positions:
        raise SystemExit("no valid positions to steer")
    print(f"design={design_id}  binder_len={len(binder)}  "
          f"steering {len(positions)} positions: {positions}", flush=True)

    tokenizer = EsmcTokenizer.from_pretrained(MODEL)
    model = EsmcForMaskedLM.from_pretrained(MODEL).to(device).eval()
    blocks = model.esmc.layers
    n_blocks = len(blocks)
    hook_idx = LAYER - 1  # hidden_states[k] is the output of layers[k-1]
    print(f"{n_blocks} ESMC blocks; hooking layers[{hook_idx}] for layer {LAYER}", flush=True)

    enc = tokenizer([binder + CHAIN_BREAK + target], return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # ---- verify the hook lands on hidden_states[LAYER] before trusting it ----
    captured = {}

    def capture(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    h = blocks[hook_idx].register_forward_hook(capture)
    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    h.remove()
    hs = ref.hidden_states[LAYER]
    if not torch.allclose(captured["h"], hs, atol=1e-3):
        raise SystemExit(
            f"hook on layers[{hook_idx}] does not match hidden_states[{LAYER}] -- "
            "layer indexing is wrong, refusing to inject into the wrong layer")
    print(f"hook verified == hidden_states[{LAYER}]", flush=True)

    # ---- direction, in RAW activation space ----
    w_dec = sae.w_dec.detach()
    def raw_dir(vec: torch.Tensor) -> torch.Tensor:
        # w_dec rows are unit-norm in centered+scaled space; convert to raw.
        return vec / scale

    dirs: dict[str, torch.Tensor] = {"feature": raw_dir(w_dec[args.feature])}
    if args.controls:
        other = args.other_feature
        if other is None:
            # A different real feature, chosen far from this one in index space.
            other = (args.feature + d_hidden // 2) % d_hidden
        dirs[f"other_feat{other}"] = raw_dir(w_dec[other])
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        for s in range(args.n_random):
            v = torch.randn(w_dec.shape[1], generator=g)
            dirs[f"random{s}"] = raw_dir((v / v.norm()).to(device))

    # Mask ONLY the probe positions; injection happens at the firing position,
    # which stays unmasked so the model still sees the motif's context.
    mask_id = model.config.mask_token_id
    inject_pos = [int(x) for x in args.inject_positions.split(",")] if args.inject_positions \
        else [fire_pos if fire_pos else positions[0]]
    probe_pos = [int(x) for x in args.probe_positions.split(",")] if args.probe_positions \
        else [q for q in (inject_pos[0] + 1, inject_pos[0] + 2) if 1 <= q <= len(binder)]
    print(f"injecting at binder positions {inject_pos}; "
          f"probing (masked) {probe_pos} -> native "
          f"{''.join(binder[q-1] for q in probe_pos)}", flush=True)

    masked_ids = input_ids.clone()
    for q in probe_pos:
        masked_ids[0, q] = mask_id

    alphas = [float(a) for a in args.alphas.split(",")]
    aa_lab = aa_labels(tokenizer)
    rows = []

    for name, d in dirs.items():
        for alpha in alphas:
            delta = (alpha * d).to(device)
            meas: dict = {}

            def inject(_m, _i, out, delta=delta, meas=meas):
                t = out[0] if isinstance(out, tuple) else out
                t = t.clone()
                for q in inject_pos:
                    t[0, q] = t[0, q] + delta
                # Measured here, on the tensor actually passed downstream --
                # see the docstring on why hidden_states cannot be used.
                z = sae.encode((t[0] - mean) * scale)
                meas["feat_act"] = float(z[inject_pos, args.feature].mean())
                meas["h_norm"] = float(t[0, inject_pos].norm(dim=-1).mean())
                return (t,) + out[1:] if isinstance(out, tuple) else t

            hh = blocks[hook_idx].register_forward_hook(inject)
            with torch.no_grad():
                out = model(input_ids=masked_ids, attention_mask=attn)
            hh.remove()

            probs = out.logits[0, :, AA_SLICE].softmax(-1)
            for q in probe_pos:
                pr = probs[q].float().cpu().numpy()
                top = np.argsort(-pr)[:3]
                rows.append({
                    "direction": name, "alpha": alpha, "probe_position": q,
                    "native_residue": binder[q - 1],
                    "p_native": round(float(pr[aa_lab.index(binder[q - 1])]), 5)
                                if binder[q - 1] in aa_lab else None,
                    "top1": aa_lab[top[0]], "top1_p": round(float(pr[top[0]]), 4),
                    "top2": aa_lab[top[1]], "top2_p": round(float(pr[top[1]]), 4),
                    "top3": aa_lab[top[2]], "top3_p": round(float(pr[top[2]]), 4),
                    "feature_activation_at_inject": round(meas["feat_act"], 3),
                    "h_norm_at_inject": round(meas["h_norm"], 1),
                    **{f"p_{a}": round(float(pr[i]), 5) for i, a in enumerate(aa_lab)},
                })

    df = pd.DataFrame(rows)
    tag = f"feature{args.feature}_{design_id}"
    df.to_csv(args.output_dir / f"steer_{tag}.csv", index=False)

    # ---- summary: per-amino-acid change vs alpha=0, feature vs controls ----
    pcols = [c for c in df.columns if c.startswith("p_") and c != "p_native"]
    summ = []
    for (name, alpha), g in df.groupby(["direction", "alpha"]):
        b0 = df[(df.direction == name) & (df.alpha == 0)].set_index("probe_position")
        g = g.set_index("probe_position")
        common = b0.index.intersection(g.index)
        for q in common:
            delta_p = g.loc[q, pcols].astype(float) - b0.loc[q, pcols].astype(float)
            gain = delta_p.sort_values(ascending=False)
            summ.append({
                "direction": name, "alpha": alpha, "probe_position": q,
                "native": g.loc[q, "native_residue"],
                "feature_act": g.loc[q, "feature_activation_at_inject"],
                "top_gainer": gain.index[0].removeprefix("p_"),
                "top_gain": round(float(gain.iloc[0]), 4),
                "top_loser": gain.index[-1].removeprefix("p_"),
                "top_loss": round(float(gain.iloc[-1]), 4),
                **{f"d_{c.removeprefix('p_')}": round(float(delta_p[c]), 4) for c in pcols},
            })
    sdf = pd.DataFrame(summ)
    sdf.to_csv(args.output_dir / f"steer_{tag}_summary.csv", index=False)

    show = ["direction", "alpha", "probe_position", "native", "feature_act",
            "top_gainer", "top_gain", "top_loser", "top_loss"]
    print("\n=== amino acid probability change vs alpha=0, at masked probe positions ===", flush=True)
    print(sdf[show].to_string(index=False), flush=True)
    print(f"\nwrote {args.output_dir}/steer_{tag}*.csv", flush=True)

    (args.output_dir / f"steer_feature{args.feature}_{design_id}_meta.json").write_text(json.dumps({
        "feature": args.feature, "design_id": design_id, "binder": binder,
        "positions": positions, "alphas": alphas, "scale": scale,
        "layer": LAYER, "hook_index": hook_idx, "n_blocks": n_blocks,
        "directions": list(dirs), "paired_input": True,
    }, indent=2))


if __name__ == "__main__":
    main()
