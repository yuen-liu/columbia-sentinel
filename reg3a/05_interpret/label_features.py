"""Draft a one-line label per SAE feature from its max-activating interface
residues, using Claude Haiku via the Message Batches API.

Evidence per example (all from interface_activations.py):
  - sequence context, +/- 8 residues, firing residue bracketed
  - which TARGET residues that binder residue contacts (<=5 A), by name+number
  - distance to target, per-residue pLDDT, burial-neighbour count, helix proxy
Plus per-feature aggregates: interface enrichment, support, overall density.

Why the Batches API: 50% cheaper and this is a bulk non-interactive job.
~$0.05 for a few dozen features.

Every label is a HYPOTHESIS from sequence + local structure, not a validated
finding. Treat it as a competent first-pass guess, and confirm anything load-
bearing against structure before relying on it. The prompt explicitly licenses
"No clear pattern" so the model is not pushed into confabulating a story --
iGEM's run showed Haiku does use that escape hatch correctly when evidence is
genuinely scattered.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python3 05_interpret/label_features.py \
        --examples-csv results/interface/top_examples.csv \
        --summary-csv results/interface/feature_summary.csv \
        --output-dir results/interface
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

MODEL = "claude-haiku-4-5-20251001"

SYSTEM = """You are analyzing features from a sparse autoencoder trained on \
ESM-C protein language model embeddings. The proteins are de novo designed \
binders (50-139 residues) against the human target Reg3A (138 residues).

Each feature is one direction in the SAE dictionary. You are shown the \
residues where it fires HARDEST, restricted to residues at the binder-target \
binding interface (within 5A of the target).

Your job: say what the feature detects, in one sentence.

Evidence notes:
- The binder residue is in [brackets]. Flanking residues are its sequence context.
- "contacts" lists the TARGET residues it touches. Several examples converging \
on the same target residue is strong evidence the feature is tied to a specific \
binding site.
- pLDDT is local structure confidence, 0-100. Below ~70 the predicted geometry \
is unreliable, so discount contact evidence from those examples.
- These are de novo designs with no evolutionary relatives, so do NOT reach for \
protein family or domain names. Describe the physical/chemical/structural \
pattern you can actually see.

If the evidence is genuinely scattered, reply exactly "No clear pattern" for \
the label rather than inventing one. That is a useful and expected answer.

Reply ONLY with JSON:
{"label": "<one sentence>", "confidence": "high"|"medium"|"low", "basis": "<which evidence drove it>"}"""


def build_prompt(feature: int, ex: pd.DataFrame, meta: dict) -> str:
    lines = [
        f"Feature {feature}",
        f"  interface enrichment (fires at contact vs away): {meta['enrichment']:.2f}x",
        f"  support: fires at {meta['support']:,} contact residues "
        f"(overall firing density {meta['density']:.4f})",
        f"  examples below are from {ex['design_id'].nunique()} distinct designs",
        "",
        "Max-activating interface residues:",
    ]
    for _, r in ex.iterrows():
        lines.append(
            f"  act={r['activation']:.2f}  {r['context']}  "
            f"contacts={r['target_contacts'] or 'none'}  "
            f"dist={r['min_dist_to_target']}A  pLDDT={r['plddt']}  "
            f"burial={r['burial_neighbors_10A']}  helix_like={r['helix_like']}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples-csv", type=Path, required=True)
    ap.add_argument("--summary-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--features", default=None, help="Comma-separated subset")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-wait-min", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print one rendered prompt and the request count, send nothing.")
    args = ap.parse_args()

    ex_all = pd.read_csv(args.examples_csv)
    summ = pd.read_csv(args.summary_csv).set_index("feature")
    feats = ([int(x) for x in args.features.split(",")] if args.features
             else sorted(ex_all["feature"].unique().tolist()))

    requests = []
    for f in feats:
        ex = ex_all[ex_all["feature"] == f].sort_values("activation", ascending=False)
        if ex.empty:
            continue
        meta = {
            "enrichment": float(summ.loc[f, "enrichment_contact_vs_far"]),
            "support": int(summ.loc[f, "n_contact_residues_fired"]),
            "density": float(summ.loc[f, "density_all"]),
        }
        requests.append({
            "custom_id": f"feature_{f}",
            "params": {
                "model": args.model,
                "max_tokens": 400,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": build_prompt(f, ex, meta)}],
            },
        })

    print(f"{len(requests)} features to label with {args.model}", flush=True)
    if args.dry_run:
        print("\n--- example prompt ---\n")
        print(requests[0]["params"]["messages"][0]["content"])
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Note that a Claude.ai Pro/Max "
                 "subscription does NOT cover programmatic API use -- this needs "
                 "an API key with billing enabled.")

    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"batch {batch.id} submitted; polling", flush=True)

    deadline = time.time() + args.max_wait_min * 60
    while time.time() < deadline:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"  {batch.processing_status} {batch.request_counts}", flush=True)
        time.sleep(20)
    else:
        sys.exit(f"batch {batch.id} did not finish within {args.max_wait_min} min; "
                 f"it is still running -- retrieve it later with its id")

    rows = []
    for res in client.messages.batches.results(batch.id):
        fid = int(res.custom_id.split("_")[1])
        if res.result.type != "succeeded":
            rows.append({"feature": fid, "label": None, "confidence": None,
                         "basis": None, "error": res.result.type})
            continue
        text = "".join(b.text for b in res.result.message.content if b.type == "text").strip()
        try:
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            parsed = json.loads(text)
            rows.append({"feature": fid, "label": parsed.get("label"),
                         "confidence": parsed.get("confidence"),
                         "basis": parsed.get("basis"), "error": None})
        except Exception:  # noqa: BLE001 - keep the raw text rather than lose it
            rows.append({"feature": fid, "label": text[:400], "confidence": None,
                         "basis": None, "error": "unparsed_json"})

    out = pd.DataFrame(rows).sort_values("feature")
    out = out.merge(
        summ[["enrichment_contact_vs_far", "n_contact_residues_fired",
              "max_act_contact", "density_all"]],
        left_on="feature", right_index=True, how="left")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "feature_labels.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} labels, "
          f"{int(out['error'].notna().sum())} errors)", flush=True)
    for _, r in out.iterrows():
        print(f"  {r['feature']:>6}  [{r['confidence'] or '?':>6}]  {r['label']}", flush=True)


if __name__ == "__main__":
    main()
