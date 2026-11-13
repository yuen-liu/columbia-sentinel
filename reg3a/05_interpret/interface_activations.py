"""Find which SAE features fire at the binder-target interface, and pull
labeling evidence for the strongest ones.

Geometry is ported from igemmodel's extract_interface_features.py: minimum
heavy-atom distance from each binder residue to any target-chain atom, with
    contact: <= --contact-cutoff   (default 5.0 A, the CAPRI/PISA convention)
    shell:   contact < d <= --shell-cutoff (default 10.0 A)
computed from each design's own predicted coordinates. Deliberately NOT a
PAE-based criterion: PAE measures the model's *confidence* about a residue
pair, which is orthogonal to whether they are actually close in space.

Residue <-> activation-row alignment is by ascending residue number, then
HARD-VERIFIED: the binder chain's residues must spell exactly the manifest
sequence for that design. A mismatch skips the design with a recorded reason
rather than silently pairing residues with the wrong SAE codes.

Two passes, because keeping every contact residue's sparse code in memory
would be ~50M entries:
    pass 1  per-feature aggregate stats over contact residues
    pass 2  for the top --top-features only, the max-activating examples
            plus the evidence needed to label them

RANKING CAVEAT, surfaced as data rather than hidden: ranking purely by
activation magnitude tends to surface the highest-density, most generic
features (iGEM's single most generic feature had density 0.76). So the
summary reports `density_all` and `enrichment` next to the activation
columns. Enrichment here is an *annotation*, never a gate -- this corpus
plausibly contains several valid binding modes, and a feature causal within
one mode is diluted against every residue of every other mode.

Usage:
    python3 05_interpret/interface_activations.py \
        --checkpoint 03_train/checkpoints/reg3a_layer23_paired/best.pt \
        --data-dir data/reg3a_layer23_paired \
        --manifest results/manifest_combined.csv \
        --min-iptm 0.7 --top-features 20 \
        --output-dir results/interface
"""

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_train"))
from sae_model import SparseAutoencoder  # noqa: E402

CONTACT, SHELL = "contact", "shell"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# --------------------------------------------------------------------------
# Structure parsing: hand-rolled mmCIF/PDB heavy-atom reader, matching the
# family of parsers already used by ipsae.py / compute_ipae.py (no biopython).
# --------------------------------------------------------------------------

def parse_cif_atoms(path: Path) -> list[dict]:
    field_idx: dict[str, int] = {}
    field_num = 0
    atoms: list[dict] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                field_idx[line.strip().split(".")[1]] = field_num
                field_num += 1
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            atom_name = parts[field_idx["label_atom_id"]]
            if atom_name[:1] == "H":
                continue
            seq_id = parts[field_idx["label_seq_id"]]
            if seq_id == ".":
                continue  # ligand atom, no residue number
            chain_f = "auth_asym_id" if "auth_asym_id" in field_idx else "label_asym_id"
            rec = {
                "chain": parts[field_idx[chain_f]],
                "resnum": int(seq_id),
                "resname": parts[field_idx["label_comp_id"]],
                "atom_name": atom_name,
                "x": float(parts[field_idx["Cartn_x"]]),
                "y": float(parts[field_idx["Cartn_y"]]),
                "z": float(parts[field_idx["Cartn_z"]]),
            }
            if "B_iso_or_equiv" in field_idx:
                try:
                    rec["bfac"] = float(parts[field_idx["B_iso_or_equiv"]])
                except ValueError:
                    rec["bfac"] = float("nan")
            atoms.append(rec)
    return atoms


def parse_pdb_atoms(path: Path) -> list[dict]:
    atoms = []
    with open(path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip()
            if resname == "LIG":
                continue
            atom_name = line[12:16].strip()
            if atom_name[:1] == "H":
                continue
            atoms.append({
                "chain": line[21].strip(), "resnum": int(line[22:26]),
                "resname": resname, "atom_name": atom_name,
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                "bfac": float(line[60:66]) if line[60:66].strip() else float("nan"),
            })
    return atoms


def parse_structure(path: Path) -> list[dict]:
    return parse_cif_atoms(path) if path.suffix == ".cif" else parse_pdb_atoms(path)


# --------------------------------------------------------------------------
# Interface classification (runs in worker processes, numpy only)
# --------------------------------------------------------------------------

def classify_design(
    design_id: str, structure_path: str, expected_sequence: str,
    target_chain: str, binder_chain: str, contact_cutoff: float, shell_cutoff: float,
) -> dict:
    """One entry per binder residue, in ascending resnum order, aligned 1:1
    with `expected_sequence` and therefore with this design's activation rows.

    Returns a dict with tiers/min_dist/plddt/contact partners, or an `error`.
    """
    try:
        atoms = parse_structure(Path(structure_path))
    except Exception as e:  # noqa: BLE001 - report, don't crash the pool
        return {"id": design_id, "error": f"parse failed: {type(e).__name__}: {e}"}

    tgt = [a for a in atoms if a["chain"] == target_chain]
    if not tgt:
        return {"id": design_id, "error": f"no atoms for target chain {target_chain!r}"}
    tgt_xyz = np.array([[a["x"], a["y"], a["z"]] for a in tgt])
    tgt_res = np.array([a["resnum"] for a in tgt])
    tgt_name = {a["resnum"]: a["resname"] for a in tgt}

    binder: dict[int, dict] = {}
    for a in atoms:
        if a["chain"] != binder_chain:
            continue
        r = binder.setdefault(a["resnum"], {"resname": a["resname"], "xyz": [], "b": []})
        r["xyz"].append((a["x"], a["y"], a["z"]))
        r["b"].append(a.get("bfac", float("nan")))
    if not binder:
        return {"id": design_id, "error": f"no atoms for binder chain {binder_chain!r}"}

    resnums = sorted(binder)
    seq = "".join(THREE_TO_ONE.get(binder[r]["resname"], "X") for r in resnums)
    if seq != expected_sequence:
        return {"id": design_id, "error": (
            f"structure binder sequence ({len(seq)} res) != manifest sequence "
            f"({len(expected_sequence)} res); skipping to avoid misaligned codes")}

    # CA coords for the cheap burial / helix proxies below.
    ca = {}
    for a in atoms:
        if a["chain"] == binder_chain and a["atom_name"] == "CA":
            ca[a["resnum"]] = np.array([a["x"], a["y"], a["z"]])
    ca_arr = np.array([ca[r] for r in resnums]) if len(ca) == len(resnums) else None

    tiers, dists, plddt, partners, burial, helixish = [], [], [], [], [], []
    for i, r in enumerate(resnums):
        xyz = np.array(binder[r]["xyz"])
        d = np.sqrt(((xyz[:, None, :] - tgt_xyz[None, :, :]) ** 2).sum(-1))
        dmin = float(d.min())
        tiers.append(CONTACT if dmin <= contact_cutoff
                     else SHELL if dmin <= shell_cutoff else None)
        dists.append(round(dmin, 2))
        plddt.append(round(float(np.nanmean(binder[r]["b"])), 2))

        # Which target residues this binder residue touches (<= contact cutoff).
        near = np.unique(tgt_res[(d <= contact_cutoff).any(axis=0)])
        partners.append([f"{tgt_name[int(t)]}{int(t)}" for t in near])

        # Burial proxy: count of OTHER binder heavy atoms within 10 A of this
        # residue. Cheap stand-in for solvent accessibility -- no new deps.
        if ca_arr is not None:
            burial.append(int((np.linalg.norm(ca_arr - ca_arr[i], axis=1) <= 10.0).sum() - 1))
            # Helix proxy: CA(i)->CA(i+4) is ~6.2 A in an alpha helix.
            if i + 4 < len(resnums):
                helixish.append(bool(abs(float(np.linalg.norm(ca_arr[i + 4] - ca_arr[i])) - 6.2) < 1.3))
            else:
                helixish.append(False)
        else:
            burial.append(-1)
            helixish.append(False)

    return {"id": design_id, "resnums": resnums, "tiers": tiers, "min_dist": dists,
            "plddt": plddt, "partners": partners, "burial": burial,
            "helixish": helixish, "sequence": seq, "error": None}


# --------------------------------------------------------------------------

def load_sae(checkpoint: Path, device: torch.device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(**ck["config"]).to(device).eval()
    sae.load_state_dict(ck["model_state_dict"])
    return sae, ck["mean"].to(device), float(ck["scale"]), ck["config"]["d_hidden"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--contact-cutoff", type=float, default=5.0)
    ap.add_argument("--shell-cutoff", type=float, default=10.0)
    ap.add_argument("--min-iptm", type=float, default=0.7)
    ap.add_argument("--top-features", type=int, default=20)
    ap.add_argument("--min-support-residues", type=int, default=30,
                    help="A feature must fire on at least this many contact "
                         "residues to be rankable. Without it, the top of the "
                         "list is single-outlier features (observed: max_act 40.8 "
                         "on n=2 residues) that cannot be labeled or steered.")
    ap.add_argument("--features", default=None,
                    help="Comma-separated feature ids to collect examples for, "
                         "overriding --rank-by/--top-features. Used to label a UNION "
                         "of several rankings in one pass, since no single ranking "
                         "answers the question alone.")
    ap.add_argument("--rank-by", default="max_act_contact",
                    choices=["max_act_contact", "mean_act_contact_when_firing",
                             "n_contact_residues_fired", "enrichment_contact_vs_far"],
                    help="Ranking metric, applied after the support filter.")
    ap.add_argument("--examples-per-feature", type=int, default=15)
    ap.add_argument("--window", type=int, default=8, help="+/- residues of sequence context")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-residues", type=int, default=8192,
                help="Residues per SAE forward pass. The code matrix is dense "
                     "(batch x d_hidden) floats, so 8192 x 16384 x 4B = 537 MB; "
                     "65536 would be 4.3 GB and OOM on MPS.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="Recompute interface geometry even if a matching cache "
                         "exists. Geometry dominates runtime and does not depend "
                         "on the SAE or the ranking, so re-ranking reuses it.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(args.manifest)
    index = pd.read_csv(args.data_dir / "index.csv").set_index("id")
    gated = manifest[manifest["iptm"] > args.min_iptm].copy()
    gated = gated[gated["id"].isin(index.index)]
    if args.limit:
        gated = gated.head(args.limit)
    print(f"device={device}  gated designs (iptm>{args.min_iptm}): {len(gated):,}", flush=True)

    # ---- geometry, in parallel (cached: independent of SAE and ranking) ----
    cache_key = (f"iface_cache_iptm{args.min_iptm}_c{args.contact_cutoff}"
                 f"_s{args.shell_cutoff}_{args.target_chain}{args.binder_chain}"
                 f"_n{len(gated)}.pkl")
    cache_path = args.output_dir / cache_key
    results: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    if cache_path.is_file() and not args.no_cache:
        with cache_path.open("rb") as fh:
            results, skipped = pickle.load(fh)
        print(f"reusing cached geometry: {cache_path.name} "
              f"({len(results):,} designs)", flush=True)
    else:
        print("classifying interfaces...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(classify_design, r["id"], r["structure_path"], r["sequence"],
                                args.target_chain, args.binder_chain,
                                args.contact_cutoff, args.shell_cutoff)
                    for _, r in gated.iterrows()]
            for n, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                if res.get("error"):
                    skipped.append((res["id"], res["error"]))
                else:
                    results[res["id"]] = res
                if n % 2000 == 0:
                    print(f"  {n:,}/{len(futs):,}", flush=True)
        with cache_path.open("wb") as fh:
            pickle.dump((results, skipped), fh)
        print(f"cached geometry -> {cache_path.name}", flush=True)
    print(f"classified {len(results):,}; skipped {len(skipped):,}", flush=True)
    for sid, reason in skipped[:5]:
        print(f"    {sid}: {reason}", flush=True)

    if not results:
        raise SystemExit("no designs could be aligned -- check --target-chain/--binder-chain")

    acts = np.load(args.data_dir / "activations.npy", mmap_mode="r")
    sae, mean, scale, d_hidden = load_sae(args.checkpoint, device)

    # ---- pass 1: per-feature aggregates over contact residues ----
    ids = sorted(results)
    rows_contact, rows_all = [], []
    for did in ids:
        start = int(index.loc[did, "start"])
        for i, tier in enumerate(results[did]["tiers"]):
            rows_all.append(start + i)
            if tier == CONTACT:
                rows_contact.append(start + i)
    rows_contact = np.array(rows_contact, dtype=np.int64)
    rows_all = np.array(rows_all, dtype=np.int64)
    print(f"contact residues: {len(rows_contact):,} of {len(rows_all):,} "
          f"({100*len(rows_contact)/max(len(rows_all),1):.1f}%)", flush=True)

    def encode_rows(row_idx: np.ndarray):
        """Yields (chunk_row_idx, codes) for the given activation rows."""
        for s in range(0, len(row_idx), args.batch_residues):
            chunk = np.sort(row_idx[s:s + args.batch_residues])
            x = torch.from_numpy(np.asarray(acts[chunk], dtype=np.float32)).to(device)
            yield chunk, sae.encode((x - mean) * scale)

    max_c = torch.zeros(d_hidden, device=device)
    sum_c = torch.zeros(d_hidden, device=device)
    cnt_c = torch.zeros(d_hidden, device=device)
    for _, z in encode_rows(rows_contact):
        max_c = torch.maximum(max_c, z.max(dim=0).values)
        sum_c += z.sum(dim=0)
        cnt_c += (z != 0).sum(dim=0).float()

    cnt_all = torch.zeros(d_hidden, device=device)
    for _, z in encode_rows(rows_all):
        cnt_all += (z != 0).sum(dim=0).float()

    n_c, n_a = len(rows_contact), len(rows_all)
    n_far = max(n_a - n_c, 1)
    cnt_far = (cnt_all - cnt_c).clamp(min=0)
    dens_c = (cnt_c / max(n_c, 1)).cpu().numpy()
    dens_far = (cnt_far / n_far).cpu().numpy()
    summary = pd.DataFrame({
        "feature": np.arange(d_hidden),
        "max_act_contact": max_c.cpu().numpy(),
        "mean_act_contact_when_firing": (sum_c / cnt_c.clamp(min=1)).cpu().numpy(),
        "n_contact_residues_fired": cnt_c.cpu().numpy().astype(int),
        "density_contact": dens_c,
        "density_all": (cnt_all / max(n_a, 1)).cpu().numpy(),
        # Annotation only, never a filter -- see module docstring.
        "enrichment_contact_vs_far": np.where(dens_far > 0, dens_c / np.maximum(dens_far, 1e-12), np.nan),
    })
    summary["supported"] = summary["n_contact_residues_fired"] >= args.min_support_residues
    summary = summary.sort_values(args.rank_by, ascending=False)
    summary.to_csv(args.output_dir / "feature_summary.csv", index=False)
    print(f"wrote {args.output_dir/'feature_summary.csv'}", flush=True)

    n_sup = int(summary["supported"].sum())
    n_live = int((summary["n_contact_residues_fired"] > 0).sum())
    print(f"features firing at >=1 contact residue: {n_live:,}/{d_hidden:,}", flush=True)
    print(f"passing support (>={args.min_support_residues} contact residues): "
          f"{n_sup:,} -- these are the rankable ones", flush=True)
    if n_sup == 0:
        raise SystemExit("no feature meets --min-support-residues; lower it or widen the corpus")

    if args.features:
        requested = [int(x.strip()) for x in args.features.split(",")]
        sup_set = set(summary.loc[summary["supported"], "feature"].astype(int))
        top = [f for f in requested if f in sup_set]
        dropped = [f for f in requested if f not in sup_set]
        print(f"using {len(top)} explicitly requested features", flush=True)
        if dropped:
            print(f"  dropped {len(dropped)} below support threshold: {dropped}", flush=True)
    else:
        top = (summary[summary["supported"]]
               .head(args.top_features)["feature"].astype(int).tolist())
        print(f"top {len(top)} supported features by {args.rank_by}: {top}", flush=True)

    # ---- pass 2: max-activating examples for the top features only ----
    top_t = torch.tensor(top, device=device)
    row_to_loc: dict[int, tuple[str, int]] = {}
    for did in ids:
        start = int(index.loc[did, "start"])
        for i, tier in enumerate(results[did]["tiers"]):
            if tier == CONTACT:
                row_to_loc[start + i] = (did, i)

    best: dict[int, list[tuple]] = {f: [] for f in top}
    for chunk, z in encode_rows(rows_contact):
        sub = z[:, top_t].cpu().numpy()
        for j, f in enumerate(top):
            col = sub[:, j]
            nz = np.nonzero(col)[0]
            if nz.size == 0:
                continue
            k = min(args.examples_per_feature, nz.size)
            pick = nz[np.argpartition(-col[nz], k - 1)[:k]]
            best[f].extend((float(col[p]), int(chunk[p])) for p in pick)
            best[f] = sorted(best[f], key=lambda t: -t[0])[:args.examples_per_feature]

    ex_rows = []
    for f in top:
        for act, row in best[f]:
            did, i = row_to_loc[row]
            r = results[did]
            seq = r["sequence"]
            lo, hi = max(0, i - args.window), min(len(seq), i + args.window + 1)
            ex_rows.append({
                "feature": f,
                "activation": round(act, 4),
                "design_id": did,
                "position": i + 1,                       # 1-based within binder
                "residue": seq[i],
                "context": f"{seq[lo:i]}[{seq[i]}]{seq[i+1:hi]}",
                "context_plain": seq[lo:hi],             # for InterProScan submission
                "min_dist_to_target": r["min_dist"][i],
                "plddt": r["plddt"][i],
                "target_contacts": ";".join(r["partners"][i]),
                "n_target_contacts": len(r["partners"][i]),
                "burial_neighbors_10A": r["burial"][i],
                "helix_like": r["helixish"][i],
                "source": gated.loc[gated["id"] == did, "source"].iloc[0],
                "design_iptm": float(gated.loc[gated["id"] == did, "iptm"].iloc[0]),
            })
    ex = pd.DataFrame(ex_rows).sort_values(["feature", "activation"], ascending=[True, False])
    # Distinct designs among each feature's examples: a feature whose evidence
    # all comes from one design is an artifact, not a motif.
    spread = ex.groupby("feature")["design_id"].nunique().rename("n_distinct_designs_in_examples")
    ex = ex.merge(spread, on="feature", how="left")
    ex.to_csv(args.output_dir / "top_examples.csv", index=False)
    print(f"wrote {args.output_dir/'top_examples.csv'} ({len(ex)} examples)", flush=True)

    (args.output_dir / "run_meta.json").write_text(json.dumps({
        "n_gated_designs": int(len(gated)),
        "n_classified": len(results),
        "n_skipped": len(skipped),
        "skipped_examples": skipped[:20],
        "n_contact_residues": int(n_c),
        "n_binder_residues": int(n_a),
        "contact_cutoff": args.contact_cutoff, "shell_cutoff": args.shell_cutoff,
        "target_chain": args.target_chain, "binder_chain": args.binder_chain,
        "min_iptm": args.min_iptm, "top_features": top,
    }, indent=2))


if __name__ == "__main__":
    main()
