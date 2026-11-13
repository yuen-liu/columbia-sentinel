"""Embed binder sequences WITH the target present, at one ESM-C layer,
writing out ONLY the binder-span per-residue activations.

Ported from igemmodel's sae/01_embed/embed_esmc_paired.py with four changes:

1. TARGET IS A REQUIRED ARGUMENT, not a hardcoded constant. The original
   hardcoded VSNL1 (vilip1); silently embedding Reg3A binders against
   vilip1 would produce plausible-looking, meaningless activations, so
   --target-sequence / --target-fasta is mandatory with no default.
2. MPS SUPPORT. The original gated autocast on `device.type == "cuda"` and
   assumed CUDA. On Apple silicon we run fp32 compute and store fp16:
   bf16 autocast on MPS is unreliable, and the storage dtype is what
   dominates the 4.6 GB output anyway.
3. transformers 5.x CLASS NAMES. `ESMCForMaskedLM`/`ESMCTokenizer` were
   renamed `EsmcForMaskedLM`/`EsmcTokenizer` in transformers 5.x.
4. RESUMABLE. Total residue count is known up front from the manifest, so
   activations.npy is preallocated as a memmap and progress.json records
   how many designs are done. A re-run picks up where it stopped instead of
   restarting a multi-hour job.

Uses ESM-C's NATIVE chain-break token `|` (verified id 31 in
EsmcTokenizer().get_vocab()), so `binder + "|" + target` is a single literal
token sequence -- the model's own trained multi-chain convention, not a
glycine-linker approximation.

Designs are processed in LENGTH-SORTED order to minimize padding waste.
activations.npy row order therefore does NOT match manifest order; index.csv
(id, start, length) is the authoritative mapping, exactly as data.py expects.

Usage:
    python3 01_embed/embed_esmc_paired.py \
        --manifest results/manifest_combined.csv \
        --output data/reg3a_layer23_paired \
        --layer 23 --target-fasta config/reg3a_target.fasta --smoke-test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers.models.esmc.modeling_esmc import EsmcForMaskedLM
from transformers.models.esmc.tokenization_esmc import EsmcTokenizer

DEFAULT_MODEL = "biohub/ESMC-300M"
CHAIN_BREAK = "|"


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def read_target(args) -> str:
    if args.target_sequence:
        seq = args.target_sequence.strip().upper()
    else:
        lines = Path(args.target_fasta).read_text().splitlines()
        seq = "".join(l.strip() for l in lines if l and not l.startswith(">")).upper()
    if not seq:
        raise SystemExit("target sequence is empty")
    if CHAIN_BREAK in seq:
        raise SystemExit(
            f"target sequence contains {CHAIN_BREAK!r}; multi-chain targets are not "
            "supported here (the binder/target split would be ambiguous)"
        )
    return seq


@torch.no_grad()
def embed_batch(
    model, tokenizer, binders: list[str], target: str, device: torch.device, layer: int
) -> list[np.ndarray]:
    """Returns one (L_i, d_model) float16 array per binder -- ONLY the binder
    span, even though the forward pass sees binder+break+target.

    The binder is placed FIRST in the concatenation, so after the CLS token
    its residues occupy positions 1..len(binder) exactly. That invariant is
    what makes the binder span recoverable without tracking token offsets.
    """
    concatenated = [b + CHAIN_BREAK + target for b in binders]
    encoded = tokenizer(concatenated, return_tensors="pt", padding=True)
    out = model(
        input_ids=encoded["input_ids"].to(device),
        attention_mask=encoded["attention_mask"].to(device),
        output_hidden_states=True,
    )
    hidden = out.hidden_states[layer].float()
    return [
        hidden[i, 1 : 1 + len(b)].to(torch.float16).cpu().numpy()
        for i, b in enumerate(binders)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    tgt = ap.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--target-sequence", type=str)
    tgt.add_argument("--target-fasta", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-tokens", type=int, default=8192,
                    help="Token budget per batch (padded). Batch size is derived "
                         "from this and the longest sequence in the batch.")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Embed only the first 4 designs, write nothing, print shapes.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    target = read_target(args)
    device = pick_device(args.device)
    print(f"device={device}  target_len={len(target)}  layer={args.layer}", flush=True)

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    # Length-sorted so each batch has near-uniform length (padding waste is
    # pure wasted compute at ~200 tokens/seq).
    manifest = manifest.sort_values("length", kind="stable").reset_index(drop=True)

    tokenizer = EsmcTokenizer.from_pretrained(args.model)
    vocab = tokenizer.get_vocab()
    if CHAIN_BREAK not in vocab:
        raise SystemExit(f"{CHAIN_BREAK!r} not in tokenizer vocab -- cannot build paired input")
    model = EsmcForMaskedLM.from_pretrained(args.model).to(device).eval()
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    if not 0 <= args.layer <= n_layers:
        raise SystemExit(f"--layer {args.layer} out of range for {n_layers}-block model")

    if args.smoke_test:
        sub = manifest.head(4)
        acts = embed_batch(model, tokenizer, sub["sequence"].tolist(), target, device, args.layer)
        for (_, row), a in zip(sub.iterrows(), acts):
            ok = a.shape == (row["length"], d_model) and np.isfinite(a).all()
            print(f"  {row['id']}: {a.shape} expected=({row['length']}, {d_model}) finite={np.isfinite(a).all()} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit("smoke test FAILED")
        print("smoke test PASSED", flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    total_res = int(manifest["length"].sum())
    acts_path = args.output / "activations.npy"
    progress_path = args.output / "progress.json"
    index_path = args.output / "index.csv"

    # Resume: progress.json is written only after the memmap is flushed, so a
    # recorded n_done is always backed by real data on disk.
    n_done, start_offset = 0, 0
    index_rows: list[dict] = []
    if progress_path.is_file() and acts_path.is_file():
        prog = json.loads(progress_path.read_text())
        if prog.get("total_residues") == total_res and prog.get("d_model") == d_model and prog.get("layer") == args.layer:
            n_done = prog["n_done"]
            start_offset = prog["offset"]
            index_rows = pd.read_csv(index_path).to_dict("records") if index_path.is_file() else []
            print(f"resuming: {n_done:,}/{len(manifest):,} designs already embedded", flush=True)
        else:
            print("progress.json does not match this config -- starting over", flush=True)

    mode = "r+" if n_done else "w+"
    acts = np.lib.format.open_memmap(
        acts_path, mode=mode, dtype=np.float16, shape=(total_res, d_model)
    )

    manifest["_tok"] = manifest["length"] + 1 + len(target) + 2  # + break + CLS/EOS
    rows = manifest.iloc[n_done:].to_dict("records")
    offset = start_offset
    t_start = time.time()
    done = n_done

    i = 0
    while i < len(rows):
        # Grow the batch while the padded token budget allows.
        batch = [rows[i]]
        i += 1
        while i < len(rows) and len(batch) < args.max_batch:
            longest = max(r["_tok"] for r in batch + [rows[i]])
            if longest * (len(batch) + 1) > args.batch_tokens:
                break
            batch.append(rows[i])
            i += 1

        embedded = embed_batch(
            model, tokenizer, [r["sequence"] for r in batch], target, device, args.layer
        )
        for r, a in zip(batch, embedded):
            if a.shape != (r["length"], d_model):
                raise SystemExit(f"{r['id']}: got {a.shape}, expected ({r['length']}, {d_model})")
            acts[offset : offset + r["length"]] = a
            index_rows.append({"id": r["id"], "start": offset, "length": r["length"]})
            offset += r["length"]
        done += len(batch)

        if done % 2000 < len(batch) or i >= len(rows):
            acts.flush()
            pd.DataFrame(index_rows).to_csv(index_path, index=False)
            progress_path.write_text(json.dumps({
                "n_done": done, "offset": offset, "total_residues": total_res,
                "d_model": d_model, "layer": args.layer,
            }))
            rate = (done - n_done) / max(time.time() - t_start, 1e-9)
            eta = (len(manifest) - done) / max(rate, 1e-9) / 60
            print(f"  {done:,}/{len(manifest):,} designs  {rate:.1f}/s  ETA {eta:.0f} min", flush=True)

    sync(device)
    acts.flush()
    if offset != total_res:
        raise SystemExit(f"wrote {offset} residues, manifest says {total_res} -- aborting")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    manifest.drop(columns=["_tok"]).to_csv(args.output / "manifest_combined.csv", index=False)
    (args.output / "target.fasta").write_text(f">target\n{target}\n")
    print(f"done: {done:,} designs, {offset:,} residues -> {acts_path} "
          f"({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
