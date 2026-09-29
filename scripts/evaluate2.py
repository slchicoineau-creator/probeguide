"""ProbeGuide robustness: (1) label-budget curve — linear probe trained on only N labeled train-fold tokens
(balanced ink/non-ink, 5 random draws) vs the incumbent (256 expert clicks); (2) small MLP probe; (3) paired
segment-level bootstrap CI of the AUPRC/AUROC difference probe - incumbent on held-out tokens."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

rng = np.random.default_rng(0)
C = []
for f in sorted(glob.glob("emb/chunk_*.npz")):
    d = np.load(f); sup = d["sup"] > 0
    e = d["emb"][sup].astype(np.float32); e /= np.linalg.norm(e, axis=1, keepdims=True)
    C.append(dict(seg=str(d["seg"]), kind=str(d["kind"]), ink=d["ink"][sup].astype(int), emb=e, inc=d["incumbent"][sup]))
segs = sorted({c["seg"] for c in C}); folds = [segs[i::4] for i in range(4)]

budget_rows, pooled = [], {"y": [], "seg": [], "inc": [], "lin": [], "mlp": [], "b256": []}
for fi, test in enumerate(folds):
    tr = [c for c in C if c["seg"] not in test]; te = [c for c in C if c["seg"] in test]
    X = np.concatenate([c["emb"] for c in tr]); y = np.concatenate([c["ink"] for c in tr])
    Xt = np.concatenate([c["emb"] for c in te]); yt = np.concatenate([c["ink"] for c in te])
    inc = np.concatenate([c["inc"] for c in te])
    for N in [32, 64, 128, 256, 512, 1024, 2048, len(y)]:
        aps, aus = [], []
        for rep in range(5 if N < len(y) else 1):
            if N < len(y):
                pi, ni = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
                idx = np.r_[rng.choice(pi, N // 2, replace=False), rng.choice(ni, N // 2, replace=False)]
            else:
                idx = np.arange(len(y))
            m = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X[idx], y[idx])
            s = m.predict_proba(Xt)[:, 1]; aps.append(average_precision_score(yt, s)); aus.append(roc_auc_score(yt, s))
            if N == 256 and rep == 0:
                b256 = s
            if N == len(y):
                lin = s
        budget_rows.append(dict(fold=fi, N=int(N), auprc=float(np.mean(aps)), auroc=float(np.mean(aus)),
                                inc_auprc=float(average_precision_score(yt, inc)), inc_auroc=float(roc_auc_score(yt, inc))))
    mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300, early_stopping=True, random_state=0).fit(X, y)
    pooled["mlp"].append(mlp.predict_proba(Xt)[:, 1]); pooled["lin"].append(lin); pooled["b256"].append(b256)
    pooled["inc"].append(inc); pooled["y"].append(yt)
    pooled["seg"].append(np.concatenate([np.full(len(c["ink"]), c["seg"]) for c in te]))
    print("fold", fi, "done", flush=True)

print("\nLabel-budget curve (mean over folds): N labeled tokens -> AUPRC / AUROC   [incumbent prototype = mean of 128 unique expert clicks]")
for N in sorted({r["N"] for r in budget_rows if r["N"] <= 2048}) + ["all"]:
    rs = [r for r in budget_rows if (r["N"] == N if N != "all" else r["N"] > 2048)]
    print(f"  N={str(N):>5s}: AUPRC {np.mean([r['auprc'] for r in rs]):.3f}  AUROC {np.mean([r['auroc'] for r in rs]):.3f}")
print(f"  incumbent: AUPRC {np.mean([r['inc_auprc'] for r in budget_rows if r['N'] == budget_rows[0]['N']]):.3f}")

y = np.concatenate(pooled["y"]); sg = np.concatenate(pooled["seg"])
arms = {k: np.concatenate(pooled[k]) for k in ["inc", "b256", "lin", "mlp"]}
print("\nPooled held-out tokens:", len(y))
for k, s in arms.items():
    print(f"  {k:5s} AUPRC {average_precision_score(y, s):.3f}  AUROC {roc_auc_score(y, s):.3f}")
useg = np.unique(sg); boot = {k: [] for k in ["b256", "lin", "mlp"]}
for _ in range(2000):
    pick = rng.choice(useg, len(useg)); idx = np.concatenate([np.nonzero(sg == s)[0] for s in pick])
    if y[idx].min() == y[idx].max():
        continue
    base = average_precision_score(y[idx], arms["inc"][idx])
    for k in boot:
        boot[k].append(average_precision_score(y[idx], arms[k][idx]) - base)
print("\nSegment-bootstrap 95% CI of AUPRC gain over the incumbent:")
for k, v in boot.items():
    print(f"  {k:5s} gain {np.mean(v):+.3f}  CI [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")
json.dump(dict(budget=budget_rows, boot={k: [float(np.percentile(v, 2.5)), float(np.mean(v)), float(np.percentile(v, 97.5))]
                                          for k, v in boot.items()}), open("results2.json", "w"), indent=1)
