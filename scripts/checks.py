"""Advisor checks 1-3: (1) within-chunk vs pooled AUROC; (2) FP rate on ink-poor chunks at MATCHED recall
(thresholds set on train folds to hit the incumbent's train recall); (3) cross-segment leakage: test chunks with
train-fold label points within 64 vx -> flag and re-score without them."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial import cKDTree

C = []
for f in sorted(glob.glob("emb/chunk_*.npz")):
    d = np.load(f); sup = d["sup"] > 0
    e = d["emb"][sup].astype(np.float32); e /= np.linalg.norm(e, axis=1, keepdims=True)
    C.append(dict(seg=str(d["seg"]), kind=str(d["kind"]), ink=d["ink"][sup].astype(int), emb=e, inc=d["incumbent"][sup],
                  raw=d["raw_cos"][sup], origin=d["origin_xyz"].astype(float)))
segs = sorted({c["seg"] for c in C}); folds = [segs[i::4] for i in range(4)]
pts = {s: np.load(f"points_{s}.npz")["xyz"] for s in segs}
trees = {s: cKDTree(pts[s][::5]) for s in segs}

rows = []
for fi, test in enumerate(folds):
    tr = [c for c in C if c["seg"] not in test]; te = [c for c in C if c["seg"] in test]
    X = np.concatenate([c["emb"] for c in tr]); y = np.concatenate([c["ink"] for c in tr])
    m = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X, y)
    for c in tr + te:
        c["probe"] = m.predict_proba(c["emb"])[:, 1]
    # matched recall: incumbent at 0.5 on train folds -> recall r*; probe threshold with same train recall
    inc_tr = np.concatenate([c["inc"] for c in tr]); pr_tr = np.concatenate([c["probe"] for c in tr])
    r_star = ((inc_tr > 0.5) & (y == 1)).sum() / (y == 1).sum()
    t_pr = np.quantile(pr_tr[y == 1], 1 - r_star)
    raw_tr = np.concatenate([c["raw"] for c in tr]); t_raw = np.quantile(raw_tr[y == 1], 1 - r_star)
    for c in te:
        # leakage: min distance from this chunk's centre region to any train-fold label point
        centre = c["origin"] + 128
        dmin = min(trees[s].query(centre)[0] for s in segs if s not in test)
        rows.append(dict(fold=fi, seg=c["seg"], kind=c["kind"], y=c["ink"], inc=c["inc"], raw=c["raw"], probe=c["probe"],
                         t_pr=t_pr, t_raw=t_raw, r_star=r_star, dmin=dmin))

def within(k, rs):
    v = [roc_auc_score(r["y"], r[k]) for r in rs if 0 < r["y"].mean() < 1]
    return np.mean(v), len(v)
print("(1) mean WITHIN-chunk AUROC (chunks with both classes):")
for k in ["inc", "raw", "probe"]:
    a, n = within(k, rows); print(f"    {k:6s} {a:.3f}  (n={n} chunks)")
print("(2) matched recall (train-fold incumbent recall r* =", np.round(np.mean([r["r_star"] for r in rows]), 3), ")")
for k, thr in [("inc", lambda r: 0.5), ("raw", lambda r: r["t_raw"]), ("probe", lambda r: r["t_pr"])]:
    y = np.concatenate([r["y"] for r in rows]); p = np.concatenate([r[k] > thr(r) for r in rows])
    poor = np.concatenate([np.full(len(r["y"]), r["kind"] == "poor") for r in rows])
    print(f"    {k:6s} held-out recall {p[y == 1].mean():.3f}  precision {(p & (y == 1)).sum() / max(p.sum(), 1):.3f}  "
          f"FP on ink-poor chunks {p[poor & (y == 0)].mean():.3f}  FP overall {p[y == 0].mean():.3f}")
print("(3) leakage: test-chunk centre distance to nearest train-fold label point (vx): min",
      round(min(r["dmin"] for r in rows)), "median", round(np.median([r["dmin"] for r in rows])))
clean = [r for r in rows if r["dmin"] > 64 + 181]   # chunk half-diagonal ~181 vx + 64 margin
print(f"    chunks kept after removing any within 64 vx of train labels: {len(clean)}/{len(rows)}")
if clean:
    y = np.concatenate([r["y"] for r in clean])
    for k in ["inc", "probe"]:
        s = np.concatenate([r[k] for r in clean])
        print(f"    {k:6s} AUPRC {average_precision_score(y, s):.3f}  AUROC {roc_auc_score(y, s):.3f}")

print("(2b) FP at held-out-matched recall (reporting only: threshold picked so held-out recall = incumbent's):")
y = np.concatenate([r["y"] for r in rows]); poor = np.concatenate([np.full(len(r["y"]), r["kind"] == "poor") for r in rows])
inc = np.concatenate([r["inc"] for r in rows]); pr = np.concatenate([r["probe"] for r in rows])
rec = (inc[y == 1] > 0.5).mean(); t = np.quantile(pr[y == 1], 1 - rec)
for k, s, th in [("inc", inc, 0.5), ("probe", pr, t)]:
    p = s > th
    print(f"    {k:6s} recall {p[y == 1].mean():.3f}  precision {(p & (y == 1)).sum() / p.sum():.3f}  FP ink-poor {p[poor & (y == 0)].mean():.3f}  FP overall {p[y == 0].mean():.3f}")
