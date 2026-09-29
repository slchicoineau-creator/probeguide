"""Probe variant for release: supervised surface tokens + 'air' negatives (tokens whose normalised CT mean is in the
lowest 15% of their chunk). Held-out comparison vs the surface-only probe (component level), plus how each behaves on
air tokens of held-out chunks (fraction of air tokens scored > 0.5)."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
rng = np.random.default_rng(0); C = []
for f in sorted(glob.glob("emb/chunk_*.npz")):
    d = np.load(f); s = d["sup"] > 0; it = np.load(f.replace("chunk_", "int_").replace(".npz", ".npy"))[..., 0]
    E = d["emb"].astype(np.float32); E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9
    air = (it < np.quantile(it, 0.15)) & ~s; ai = np.argwhere(air); ai = ai[rng.choice(len(ai), min(300, len(ai)), replace=False)]
    C.append(dict(seg=str(d["seg"]), emb=E[s], ink=d["ink"][s], air=E[tuple(ai.T)], inc=d["incumbent"][s],
                  inc_air=d["incumbent"][tuple(ai.T)]))
segs = sorted({c["seg"] for c in C}); folds = [segs[i::4] for i in range(4)]
out = {"surface_only": [], "with_air": []}; air_hi = {"incumbent": [], "surface_only": [], "with_air": []}
for test in folds:
    tr = [c for c in C if c["seg"] not in test]; te = [c for c in C if c["seg"] in test]
    X = np.concatenate([c["emb"] for c in tr]); y = np.concatenate([c["ink"] for c in tr]); A = np.concatenate([c["air"] for c in tr])
    m1 = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X, y)
    m2 = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(np.concatenate([X, A]), np.r_[y, np.zeros(len(A))])
    Xt = np.concatenate([c["emb"] for c in te]); yt = np.concatenate([c["ink"] for c in te]); At = np.concatenate([c["air"] for c in te])
    for k, m in [("surface_only", m1), ("with_air", m2)]:
        s = m.predict_proba(Xt)[:, 1]; out[k].append((average_precision_score(yt, s), roc_auc_score(yt, s)))
        air_hi[k].append((m.predict_proba(At)[:, 1] > 0.5).mean())
    air_hi["incumbent"].append((np.concatenate([c["inc_air"] for c in te]) > 0.5).mean())
for k, v in out.items():
    v = np.array(v); print(f"{k:13s} AUPRC {v[:,0].mean():.3f}  AUROC {v[:,1].mean():.3f}   air tokens scored >0.5: {np.mean(air_hi[k]):.3f}")
print(f"{'incumbent':13s} air tokens scored >0.5: {np.mean(air_hi['incumbent']):.3f}")
