"""ProbeGuide evaluation: incumbent single-prototype min-max similarity vs calibrated alternatives, on held-out Paris 4
segments (4 folds x 2 segments). Token-level, supervised tokens only (tokens that contain mapped 2D label points).

Arms (score per token; higher = more ink-like):
  incumbent      exact DinoGuidedLabelGenerator similarity: blended cos to avg_ref_embedding, min-max per 256^3 chunk
  raw_cos        same blended cosine without the per-chunk min-max
  multiproto     max cosine to any of the 256 expert click tokens (recorded_embeddings*.npy)  [same label budget]
  click_probe    logistic regression: 256 click tokens (pos) vs random unlabeled train-fold tokens (neg)  [same budget]
  sup_probe      logistic regression on the train folds' supervised token labels (stronger head on frozen features)
Metrics: AUPRC, AUROC (held-out tokens), and at the generator's operating threshold (score > 0.5; probes output
probabilities): precision, recall, and false-positive rate on ink-free supervised tokens of *ink-poor* chunks.
Calibrated arms (probes, raw_cos) get their threshold chosen on the train folds (max F1); incumbent uses 0.5 as shipped.
"""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

rng = np.random.default_rng(0)
files = sorted(glob.glob("emb/chunk_*.npz"))
C = []
for f in files:
    d = np.load(f)
    sup = d["sup"] > 0
    C.append(dict(seg=str(d["seg"]), kind=str(d["kind"]), sup=sup, ink=d["ink"][sup].astype(int),
                  emb=d["emb"][sup].astype(np.float32), inc=d["incumbent"][sup], raw=d["raw_cos"][sup],
                  inten=np.load(f.replace("chunk_","int_").replace(".npz",".npy"))[sup],
                  unl=d["emb"].reshape(-1, 864)[rng.choice(32 ** 3, 300, replace=False)].astype(np.float32)))
segs = sorted({c["seg"] for c in C})
folds = [segs[i::4] for i in range(4)]
clicks = np.concatenate([np.load(f"{_env.ASSETS}/recorded_embeddings.npy"), np.load(f"{_env.ASSETS}/recorded_embeddings_2.npy")]).astype(np.float32)
clicks_n = clicks / np.linalg.norm(clicks, axis=1, keepdims=True)
print("chunks", len(C), "segments", len(segs), "supervised tokens", sum(len(c["ink"]) for c in C),
      "ink rate", round(np.mean(np.concatenate([c["ink"] for c in C])), 3))


def norm(e):
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def best_thr(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    return t[np.argmax(f[:-1])]


results = {k: [] for k in ["incumbent", "raw_cos", "multiproto", "click_probe", "intensity_probe", "sup_probe"]}
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
for fi, test in enumerate(folds):
    tr = [c for c in C if c["seg"] not in test]; te = [c for c in C if c["seg"] in test]
    Xtr = np.concatenate([norm(c["emb"]) for c in tr]); ytr = np.concatenate([c["ink"] for c in tr])
    sup = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    ip = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")).fit(
        np.concatenate([c["inten"] for c in tr]), ytr)
    U = norm(np.concatenate([c["unl"] for c in tr]))
    cp = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(
        np.concatenate([clicks_n, U]), np.r_[np.ones(len(clicks_n)), np.zeros(len(U))])
    arms = {
        "incumbent": lambda c: c["inc"],
        "raw_cos": lambda c: c["raw"],
        "multiproto": lambda c: (norm(c["emb"]) @ clicks_n.T).max(1),
        "click_probe": lambda c: cp.predict_proba(norm(c["emb"]))[:, 1],
        "intensity_probe": lambda c: ip.predict_proba(c["inten"])[:, 1],
        "sup_probe": lambda c: sup.predict_proba(norm(c["emb"]))[:, 1],
    }
    for name, fn in arms.items():
        s_tr = np.concatenate([fn(c) for c in tr]); thr = 0.5 if name == "incumbent" else best_thr(ytr, s_tr)
        s = np.concatenate([fn(c) for c in te]); y = np.concatenate([c["ink"] for c in te])
        poor = np.concatenate([np.full(len(c["ink"]), c["kind"] == "poor") for c in te])
        pred = s > thr
        results[name].append(dict(fold=fi, test=test, auprc=average_precision_score(y, s), auroc=roc_auc_score(y, s),
                                  thr=float(thr), precision=float((pred & (y == 1)).sum() / max(pred.sum(), 1)),
                                  recall=float((pred & (y == 1)).sum() / max((y == 1).sum(), 1)),
                                  fp_poor=float(pred[poor & (y == 0)].mean()) if (poor & (y == 0)).any() else np.nan,
                                  base_rate=float(y.mean())))
print(f"{'arm':12s} {'AUPRC':>13s} {'AUROC':>13s} {'prec@op':>8s} {'rec@op':>7s} {'FP ink-poor':>11s}  per-fold AUPRC")
summary = {}
for name, rs in results.items():
    a = np.array([r["auprc"] for r in rs]); b = np.array([r["auroc"] for r in rs])
    summary[name] = dict(auprc=a.mean(), auroc=b.mean(), precision=np.mean([r["precision"] for r in rs]),
                         recall=np.mean([r["recall"] for r in rs]), fp_poor=np.nanmean([r["fp_poor"] for r in rs]))
    print(f"{name:12s} {a.mean():.3f}±{a.std():.3f} {b.mean():.3f}±{b.std():.3f} {summary[name]['precision']:8.3f} "
          f"{summary[name]['recall']:7.3f} {summary[name]['fp_poor']:11.3f}  {np.round(a, 3).tolist()}")
print("base ink rate per fold:", [round(r["base_rate"], 3) for r in results["incumbent"]])
json.dump(dict(results=results, summary=summary), open("results.json", "w"), indent=1, default=float)
