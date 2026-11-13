"""Per-design contact fingerprints: which target residues each binder touches."""
import json, glob, os, numpy as np
ROOT = "/Users/vigneshkarthik/Documents/liuLab/iGem/boltz-experiments"
CUT, MIN_IPTM, NRES = 5.0, 0.7, 138

def parse(cif):
    tx, tr, bx = [], [], []
    with open(cif) as fh:
        for line in fh:
            if not line.startswith("ATOM"): continue
            f = line.split()
            if f[2] == "H": continue
            xyz = (float(f[10]), float(f[11]), float(f[12]))
            if f[15] == "A": tx.append(xyz); tr.append(int(f[7]))
            elif f[15] == "B": bx.append(xyz)
    return np.array(tx), np.array(tr), np.array(bx)

ids, camps, iptms, rows = [], [], [], []
for cdir in sorted(glob.glob(os.path.join(ROOT, "REG3A*"))):
    camp = os.path.basename(cdir)
    for m in glob.iglob(os.path.join(cdir, "results", "*", "files", "result", "metrics.json")):
        try: ip = json.load(open(m))["iptm"]
        except Exception: continue
        if ip <= MIN_IPTM: continue
        cif = glob.glob(os.path.join(os.path.dirname(m), "*_predicted.cif"))
        if not cif: continue
        tx, tr, bx = parse(cif[0])
        if len(tx) == 0 or len(bx) == 0: continue
        hit = (np.linalg.norm(bx[:, None, :] - tx[None, :, :], axis=2) <= CUT).any(axis=0)
        v = np.zeros(NRES, dtype=np.float32)
        touched = np.unique(tr[hit])
        v[touched - 1] = 1.0
        rows.append(v); ids.append(os.path.basename(cif[0]).replace("_predicted.cif", ""))
        camps.append(camp); iptms.append(ip)
X = np.vstack(rows)
np.savez_compressed("fingerprints.npz", X=X, ids=np.array(ids),
                    camp=np.array(camps), iptm=np.array(iptms))
print(f"{X.shape[0]:,} gated designs x {X.shape[1]} target residues")
print(f"mean residues contacted per design: {X.sum(1).mean():.1f}")
