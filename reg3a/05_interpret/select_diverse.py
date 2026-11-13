"""Diversity-aware candidate selection for de novo binder campaigns.

Ranking a design pool by ipTM and taking the top N concentrates the selection
on whichever binding mode happens to score highest, so most of the modes the
generator found never reach the bench. This clusters designs by *what they
actually touch on the target* and selects top-k per mode plus the most
distant outliers.

On the Reg3A corpus (12,060 designs passing ipTM > 0.7):

    naive top 29 by ipTM  ->  2 of 6 modes, 28 of 29 from one mode
    top 3 per mode + 3    ->  6 of 6 modes, 21 designs

The largest mode the naive selection never sees holds 4,460 designs whose
best member scores ipTM 0.959, against a global best of 0.985.

Usage:
    python select_diverse.py --fingerprints fingerprints.npz --k 3 --outliers 3
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def choose_k(Xn: np.ndarray, candidates=(4, 6, 8, 10, 12, 16), seed: int = 0):
    """Pick the cluster count by silhouette score on a subsample."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Xn), min(3000, len(Xn)), replace=False)
    best = None
    for k in candidates:
        km = KMeans(k, n_init=6, random_state=seed).fit(Xn)
        score = silhouette_score(Xn[sub], km.labels_[sub])
        print(f"   k={k:2d}  silhouette={score:.3f}")
        if best is None or score > best[1]:
            best = (k, score, km)
    return best


def select(labels, scores, dists, k: int, n_outliers: int):
    """Top-k per cluster by score, plus the n most distant designs overall."""
    picks: set[int] = set()
    for cl in np.unique(labels):
        idx = np.where(labels == cl)[0]
        picks.update(idx[np.argsort(-scores[idx])[:k]].tolist())
    picks.update(np.argsort(-dists)[:n_outliers].tolist())
    return sorted(picks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fingerprints", required=True,
                    help="npz with X (n x n_target_residues binary), iptm, ids")
    ap.add_argument("--k", type=int, default=3, help="designs per binding mode")
    ap.add_argument("--outliers", type=int, default=3, help="most-distant designs to add")
    ap.add_argument("--output", default="selected.csv")
    args = ap.parse_args()

    d = np.load(args.fingerprints, allow_pickle=True)
    X, ip, ids = d["X"], d["iptm"], d["ids"]
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

    print(f"{len(X):,} designs x {X.shape[1]} target residues")
    k, sil, km = choose_k(Xn)
    lab = km.labels_
    dists = np.linalg.norm(Xn - km.cluster_centers_[lab], axis=1)
    print(f"\nchosen k={k} (silhouette {sil:.3f})\n")

    for cl in np.argsort(-np.bincount(lab)):
        top = np.argsort(-km.cluster_centers_[cl])[:6] + 1
        n = int((lab == cl).sum())
        print(f"   mode {cl}: n={n:5d} ({100*n/len(lab):4.1f}%)  "
              f"median ipTM {np.median(ip[lab == cl]):.3f}  "
              f"contacts {sorted(top.tolist())}")

    naive = np.argsort(-ip)[:args.k * k + args.outliers]
    div = select(lab, ip, dists, args.k, args.outliers)
    for name, sel in (("naive top-N by ipTM", naive), ("diversity selection", div)):
        cnt = np.bincount(lab[sel], minlength=k)
        print(f"\n{name}: {len(sel)} designs, {int((cnt > 0).sum())}/{k} modes, "
              f"largest share {100*cnt.max()/len(sel):.0f}%")

    with open(args.output, "w") as fh:
        fh.write("id,mode,iptm,dist_to_centroid\n")
        for i in div:
            fh.write(f"{ids[i]},{lab[i]},{ip[i]:.4f},{dists[i]:.4f}\n")
    print(f"\nwrote {args.output} ({len(div)} designs)")


if __name__ == "__main__":
    main()
