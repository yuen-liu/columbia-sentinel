"""Build manifest_combined.csv for the Reg3A binder corpus.

Walks the Boltz design-campaign tree (REG3A*/results/<id>/metadata.json) and
emits one row per design with its binder sequence and the quality metrics
Boltz already computed. Stdlib only, so it can run before torch finishes
installing.

Output columns:
    id                      design id (pres_*)
    sequence                binder sequence (chain B)
    source                  campaign directory name
    length                  len(sequence)
    iptm                    Boltz ipTM
    binding_confidence      Boltz binding confidence
    min_interaction_pae     Boltz min interaction PAE -- NOTE: a MINIMUM, not
                            the BindCraft-convention mean ipAE that
                            scripts/compute_ipae.py computes. Useful as a
                            cheap prefilter only; the real gate is mean ipAE.
    structure_path          absolute path to the predicted .cif
    pae_path                absolute path to pae.npz

Usage:
    python3 00_data/build_manifest.py \
        --campaigns-dir ~/Documents/liuLab/iGem/boltz-experiments \
        --glob 'REG3A*' \
        --output results/manifest_combined.csv
"""

import argparse
import csv
import json
from pathlib import Path


def collect(campaigns_dir: Path, pattern: str) -> tuple[list[dict], list[tuple[str, str]]]:
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for campaign in sorted(campaigns_dir.glob(pattern)):
        results_dir = campaign / "results"
        if not results_dir.is_dir():
            skipped.append((campaign.name, "no results/ subdirectory"))
            continue

        for design_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
            meta_path = design_dir / "metadata.json"
            if not meta_path.is_file():
                skipped.append((design_dir.name, "no metadata.json"))
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                skipped.append((design_dir.name, f"unreadable metadata.json: {exc}"))
                continue

            entities = meta.get("entities") or []
            sequence = entities[0].get("value", "") if entities else ""
            if not sequence:
                skipped.append((design_dir.name, "no binder sequence in metadata"))
                continue

            # Structure + PAE are needed downstream (interface geometry, ipAE).
            # Record their paths now and flag absence rather than discovering
            # it hours later mid-run.
            result_dir = design_dir / "files" / "result"
            cifs = sorted(result_dir.glob("*_predicted.cif"))
            pae = result_dir / "pae.npz"
            if not cifs:
                skipped.append((design_dir.name, "no *_predicted.cif"))
                continue

            metrics = meta.get("metrics") or {}
            rows.append(
                {
                    "id": meta.get("id") or design_dir.name,
                    "sequence": sequence,
                    "source": campaign.name,
                    "length": len(sequence),
                    "iptm": metrics.get("iptm"),
                    "binding_confidence": metrics.get("binding_confidence"),
                    "min_interaction_pae": metrics.get("min_interaction_pae"),
                    "structure_path": str(cifs[0]),
                    "pae_path": str(pae) if pae.is_file() else "",
                }
            )

    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaigns-dir", type=Path, required=True)
    ap.add_argument("--glob", default="REG3A*")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows, skipped = collect(args.campaigns_dir.expanduser(), args.glob)
    if not rows:
        raise SystemExit(f"no designs found under {args.campaigns_dir}/{args.glob}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "sequence", "source", "length", "iptm", "binding_confidence",
        "min_interaction_pae", "structure_path", "pae_path",
    ]
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n_pae = sum(1 for r in rows if r["pae_path"])
    total_res = sum(r["length"] for r in rows)
    print(f"wrote {args.output}: {len(rows):,} designs, {total_res:,} binder residues")
    print(f"  with pae.npz: {n_pae:,}")
    print(f"  skipped: {len(skipped):,}")
    for name, reason in skipped[:5]:
        print(f"    {name}: {reason}")

    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<28} {n:,}")


if __name__ == "__main__":
    main()
