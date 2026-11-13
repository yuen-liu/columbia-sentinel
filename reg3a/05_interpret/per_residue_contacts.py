"""Per-target-residue contact counts over the gated Reg3A design corpus."""
import json, sys, glob, os
from collections import Counter
import numpy as np

ROOT = "/Users/vigneshkarthik/Documents/liuLab/iGem/boltz-experiments"
CUT, MIN_IPTM = 5.0, 0.7
TARGET, BINDER = "A", "B"

def parse(cif):
    """heavy-atom coords + residue ids, split by chain."""
    tx, bx, tr, br = [], [], [], []
    with open(cif) as fh:
        for line in fh:
            if not line.startswith("ATOM"): continue
            f = line.split()
            if f[2] == "H": continue              # type_symbol
            ch, resi = f[15], int(f[7])           # auth_asym_id, auth_seq_id
            xyz = (float(f[10]), float(f[11]), float(f[12]))
            if ch == TARGET:   tx.append(xyz); tr.append(resi)
            elif ch == BINDER: bx.append(xyz); br.append(resi)
    return (np.array(tx), np.array(tr), np.array(bx), np.array(br))

def main(campaign_glob="REG3A*", limit=None, out="contacts.json"):
    per_res = Counter()          # target residue -> contact pairs
    n_designs = n_gated = 0
    n_contact_res = n_binder_res = n_pairs = 0
    per_campaign = {}
    dirs = sorted(glob.glob(os.path.join(ROOT, campaign_glob)))
    for cdir in dirs:
        camp = os.path.basename(cdir); cres = Counter(); cpairs = 0
        for mfile in glob.iglob(os.path.join(cdir, "results", "*", "files", "result", "metrics.json")):
            n_designs += 1
            if limit and n_gated >= limit: break
            try:
                if json.load(open(mfile))["iptm"] <= MIN_IPTM: continue
            except Exception: continue
            cif = glob.glob(os.path.join(os.path.dirname(mfile), "*_predicted.cif"))
            if not cif: continue
            tx, tr, bx, br = parse(cif[0])
            if len(tx) == 0 or len(bx) == 0: continue
            n_gated += 1
            d = np.linalg.norm(bx[:, None, :] - tx[None, :, :], axis=2) <= CUT
            # unique (binder residue, target residue) pairs
            bi, ti = np.nonzero(d)
            pairs = set(zip(br[bi].tolist(), tr[ti].tolist()))
            for _, t in pairs: per_res[t] += 1; cres[t] += 1
            cpairs += len(pairs); n_pairs += len(pairs)
            n_binder_res += len(np.unique(br))
            n_contact_res += len({b for b, _ in pairs})
        per_campaign[camp] = {"pairs": cpairs, "per_res": dict(cres)}
        print(f"  {camp:28s} gated so far={n_gated:6d}  pairs={cpairs}", flush=True)
    res = {"n_designs_seen": n_designs, "n_gated": n_gated, "n_contact_pairs": n_pairs,
           "n_contact_residues": n_contact_res, "n_binder_residues": n_binder_res,
           "per_residue": dict(per_res), "per_campaign": per_campaign}
    json.dump(res, open(out, "w"))
    print(f"\ngated={n_gated}  pairs={n_pairs}  contact_res={n_contact_res}  binder_res={n_binder_res}")
    return res

if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "REG3A*"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(g, lim, sys.argv[3] if len(sys.argv) > 3 else "contacts.json")
