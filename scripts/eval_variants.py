"""S-alone comparison against the team's ACTUAL settings (CPU only, from stored stride-128 token embeddings):
  inc128      team config (`ink_detection/configs/dino_guided_v1.json`: dino_stride 128): cos to avg_ref, min-max per
              256^3 chunk, label if > 0.5. With stride 128 the 8 windows do not overlap, so the Gaussian blend is a
              no-op and the token score is exactly cos(token, ref).
  otsu128     the team's newer DINO-guidance binarisation (Open Problems page, fiber model): cos binarised per chunk
              by Otsu's method (continuous score = cos - chunk Otsu threshold).
  raw_cos     cos without any per-chunk normalisation.
  probe       ProbeGuide linear probe (fold-trained on the other segments' labels).
  probe_128   probe trained on only 128 labeled tokens (= the 128 UNIQUE expert clicks behind avg_ref: the two HF
              recorded_embeddings files are byte-identical), 5 draws.
Metrics on held-out supervised surface tokens: AUPRC, AUROC (pooled; paired segment bootstrap vs inc128), and the
binary labels each method would emit: precision, recall, F1, FP on ink-free tokens of ink-poor chunks."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, json, sys, numpy as np
DENSE = "--dense" in sys.argv   # use dense majority-vote token labels from relabel_dense.py
from skimage.filters import threshold_otsu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

rng = np.random.default_rng(0)
ref = np.load(f"{_env.ASSETS}/avg_ref_embedding.npy").astype(np.float32); ref /= np.linalg.norm(ref)
C = []
for f in sorted(glob.glob("emb/chunk_*.npz")):
    d = np.load(f); sup = d["sup"] > 0; lab = d["ink"]
    if DENSE:
        dd = np.load(f.replace("chunk_", "dense_")); sup = dd["n_sup"] > 0; lab = (dd["ink_frac"] > 0.5).astype(np.uint8)
    E = d["emb"].astype(np.float32); E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9
    cos = E @ ref                                             # (32,32,32) token cosine, stride-128 windows
    inc = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-6)
    ot = threshold_otsu(cos)
    C.append(dict(seg=str(d["seg"]), kind=str(d["kind"]), y=lab[sup].astype(int), emb=E[sup],
                  inc128=inc[sup], otsu128=cos[sup] - ot, raw_cos=cos[sup]))
segs = sorted({c["seg"] for c in C}); folds = [segs[i::4] for i in range(4)]
for fi, test in enumerate(folds):
    tr = [c for c in C if c["seg"] not in test]
    X = np.concatenate([c["emb"] for c in tr]); y = np.concatenate([c["y"] for c in tr])
    m = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X, y)
    small = []
    for rep in range(5):
        pi, ni = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
        idx = np.r_[rng.choice(pi, 64, replace=False), rng.choice(ni, 64, replace=False)]
        small.append(LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X[idx], y[idx]))
    for c in C:
        if c["seg"] in test:
            c["probe"] = m.predict_proba(c["emb"])[:, 1]
            c["probe_128"] = np.mean([s.predict_proba(c["emb"])[:, 1] for s in small], axis=0)

arms = {"inc128": 0.5, "otsu128": 0.0, "raw_cos": None, "probe": 0.5, "probe_128": 0.5}
y = np.concatenate([c["y"] for c in C]); sg = np.concatenate([np.full(len(c["y"]), c["seg"]) for c in C])
poor = np.concatenate([np.full(len(c["y"]), c["kind"] == "poor") for c in C])
out = {}
print(f"{'arm':10s} {'AUPRC':>6s} {'AUROC':>6s} | binary label: {'prec':>5s} {'recall':>6s} {'F1':>5s} {'FP ink-poor':>11s}")
for k, thr in arms.items():
    s = np.concatenate([c[k] for c in C]); row = dict(auprc=average_precision_score(y, s), auroc=roc_auc_score(y, s))
    if thr is not None:
        p = s > thr; P = (p & (y == 1)).sum() / max(p.sum(), 1); R = (p & (y == 1)).sum() / (y == 1).sum()
        row.update(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9), fp_poor=p[poor & (y == 0)].mean())
    out[k] = row
    print(f"{k:10s} {row['auprc']:6.3f} {row['auroc']:6.3f} | " + ("" if thr is None else
          f"{row['precision']:17.3f} {row['recall']:6.3f} {row['f1']:5.3f} {row['fp_poor']:11.3f}"))
useg = np.unique(sg)
for k in ["otsu128", "probe", "probe_128"]:
    s1 = np.concatenate([c[k] for c in C]); s0 = np.concatenate([c["inc128"] for c in C]); b = []
    for _ in range(2000):
        idx = np.concatenate([np.nonzero(sg == s)[0] for s in rng.choice(useg, len(useg))])
        b.append(average_precision_score(y[idx], s1[idx]) - average_precision_score(y[idx], s0[idx]))
    out[k]["auprc_gain_vs_inc128_CI"] = [float(np.percentile(b, 2.5)), float(np.mean(b)), float(np.percentile(b, 97.5))]
    print(f"AUPRC gain {k} - inc128: {np.mean(b):+.3f}  95% CI [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
# per-segment paired differences (n=8) for the probe
per = []
for s in useg:
    m_ = sg == s
    if 0 < y[m_].mean() < 1:
        per.append(average_precision_score(y[m_], np.concatenate([c["probe"] for c in C])[m_]) -
                   average_precision_score(y[m_], np.concatenate([c["inc128"] for c in C])[m_]))
print("per-segment AUPRC gain probe - inc128:", np.round(per, 3).tolist(), "positive in", sum(p > 0 for p in per), "of", len(per))
print("labels:", "dense majority vote" if DENSE else "sparse (original)", "| tokens:", len(y), "| ink rate", round(float(y.mean()), 3))
json.dump(out, open("variants_results_dense.json" if DENSE else "variants_results.json", "w"), indent=1, default=float)
