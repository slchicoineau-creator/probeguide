"""Label-QA check: where do the two independent signals (cross-scroll teacher T, never saw Paris 4; cross-segment
DINO probe, fold-trained on other segments) jointly disagree with the Paris 4 labels, and what is that disagreement?
Descriptive only (no labels are altered, no tuning). CPU, read-only."""
import glob, numpy as np, time
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import _env  # path setup; written during brainstorm round PG (agent 07)
D0 = _env.DATA
t0 = time.time()
W = np.load(f"{D0}/wt_points.npz", allow_pickle=True)
seg, ch, xyz, y = W["seg"].astype(str), W["chunk"].astype(int), W["xyz"], W["ink"].astype(int)
rr, cc, T = W["rr"], W["cc"], W["t_fwd"].astype(np.float64)
files = sorted(glob.glob(f"{D0}/emb/chunk_*.npz")); D = [np.load(f) for f in files]
DN = [np.load(f.replace("chunk_", "dense_")) for f in files]
E = []
for d in D:
    e = d["emb"].astype(np.float32); E.append(e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-9))
segs = sorted({str(d["seg"]) for d in D}); folds = [segs[i::4] for i in range(4)]
tok = np.zeros((len(y), 3), int)
for n in np.unique(ch):
    m = ch == n; tok[m] = np.clip(((xyz[m] - np.asarray(D[n]["origin_xyz"])) // 8).astype(int), 0, 31)
P = np.zeros(len(y)); IF = np.zeros(len(y))
for n in np.unique(ch):
    m = ch == n; IF[m] = DN[n]["ink_frac"][tok[m, 2], tok[m, 1], tok[m, 0]]
for test in folds:
    tr = [n for n, d in enumerate(D) if str(d["seg"]) not in test]
    X = np.concatenate([E[n][D[n]["sup"] > 0] for n in tr]); yy = np.concatenate([D[n]["ink"][D[n]["sup"] > 0] for n in tr])
    p = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X, yy)
    for n in np.unique(ch[np.isin(seg, test)]):
        m = ch == n; P[m] = p.predict_proba(E[n][tok[m, 2], tok[m, 1], tok[m, 0]])[:, 1]
print("probe done", f"{time.time()-t0:.0f}s")
# canvas distance (level-2 px) from each label-0 pixel to the nearest label-1 pixel of the same segment, and v.v.
dist = np.full(len(y), np.nan)
for s in segs:
    m = np.nonzero(seg == s)[0]; pts = np.c_[rr[m], cc[m]].astype(float); yi = y[m]
    from scipy.spatial import cKDTree
    ti, tn = cKDTree(pts[yi == 1]), cKDTree(pts[yi == 0])
    dd = np.empty(len(m)); dd[yi == 0] = ti.query(pts[yi == 0])[0]; dd[yi == 1] = tn.query(pts[yi == 1])[0]
    dist[m] = dd

from scipy import stats
from sklearn.metrics import average_precision_score as AP
ref = np.load(f"{_env.ASSETS}/avg_ref_embedding.npy").astype(np.float32); ref /= np.linalg.norm(ref)
SP = np.zeros(len(y))
for n in np.unique(ch):
    cos = E[n] @ ref; inc = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-6)
    m = ch == n; SP[m] = inc[tok[m, 2], tok[m, 1], tok[m, 0]]
arms = {"T": T, "TxS_proto": T * SP, "TxS_probe": T * P}
print("edge-band robustness: per-segment AUPRC, pixels farther than B px from the opposite label class")
for B in (0, 3, 6):
    keep = dist > B if B else np.ones(len(y), bool)
    per = {k: np.array([AP(y[(seg == s) & keep], v[(seg == s) & keep]) for s in segs]) for k, v in arms.items()}
    msg = f"B={B}: kept {keep.mean():.3f} | " + " ".join(f"{k} {v.mean():.3f}" for k, v in per.items())
    for k in ("TxS_proto", "TxS_probe"):
        d = per[k] - per["T"]; msg += f" | {k}-T {d.mean():+.3f} {(d > 0).sum()}/8 p={stats.wilcoxon(d).pvalue:.3f}"
    print(msg)
