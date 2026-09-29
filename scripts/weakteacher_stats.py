"""Analysis of the pre-registered weak-teacher test (CPU). Input: wt_points.npz from weakteacher.py + emb/chunk_*.npz.

Arms, scored at every supervised label pixel of the held-out segments (4 folds x 2 segments):
  T              teacher alone (depth direction chosen per fold on the 6 TRAINING segments)
  T*S_proto      team's prototype S (cos to avg_ref, min-max per 256^3 chunk, stride 128)
  T*S_probe128   probe fit on 128 train-fold labeled tokens (64 ink + 64 non-ink), 5 draws averaged (= click budget)
  T*S_probe      probe fit on all train-fold labeled tokens
  S_probe        probe alone (the "just use the labels" baseline)
Primary tests (per-segment AUPRC differences, n = 8; Wilcoxon + sign; AUROC secondary):
  (a) T*S_proto - T    (b) T*S_probe128 - T*S_proto    (c) T*S_probe - T*S_proto AND T*S_probe - S_probe
Teacher gate: training-segment AUROC < 0.58 in both directions -> noise (stop); >= 0.80 -> not weak."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, json, sys, numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

src = sys.argv[1] if len(sys.argv) > 1 else "wt_points.npz"
W = np.load(src, allow_pickle=True)
seg_pt, chunk_pt, xyz, y = W["seg"].astype(str), W["chunk"].astype(int), W["xyz"], W["ink"].astype(int)
ref = np.load(f"{_env.ASSETS}/avg_ref_embedding.npy").astype(np.float32); ref /= np.linalg.norm(ref)
D = [np.load(f) for f in sorted(glob.glob("emb/chunk_*.npz"))]
E = []
for d in D:
    e = d["emb"].astype(np.float32); E.append(e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-9))
segs = sorted({str(d["seg"]) for d in D}); folds = [segs[i::4] for i in range(4)]
rng = np.random.default_rng(0)

# token of every point (emb arrays are z, y, x)
tok = np.zeros((len(y), 3), int)
for n in np.unique(chunk_pt):
    m = chunk_pt == n; tok[m] = np.clip(((xyz[m] - np.asarray(D[n]["origin_xyz"])) // 8).astype(int), 0, 31)
s_proto = np.zeros(len(y), np.float32)
for n in np.unique(chunk_pt):
    cos = E[n] @ ref; inc = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-6)
    m = chunk_pt == n; s_proto[m] = inc[tok[m, 2], tok[m, 1], tok[m, 0]]

s_probe = np.zeros(len(y), np.float32); s_p128 = np.zeros(len(y), np.float32); t = np.zeros(len(y), np.float32)
report = {"teacher": {}}
for fi, test in enumerate(folds):
    tr = [n for n, d in enumerate(D) if str(d["seg"]) not in test]
    X = np.concatenate([E[n][D[n]["sup"] > 0] for n in tr]); yy = np.concatenate([D[n]["ink"][D[n]["sup"] > 0] for n in tr])
    full = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X, yy)
    small = []
    for _ in range(5):
        idx = np.r_[rng.choice(np.nonzero(yy == 1)[0], 64, replace=False), rng.choice(np.nonzero(yy == 0)[0], 64, replace=False)]
        small.append(LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X[idx], yy[idx]))
    # teacher depth direction from TRAINING segments only
    trm = ~np.isin(seg_pt, test)
    auc = {k: AUC(y[trm], W[k][trm]) for k in ("t_fwd", "t_rev")}
    pick = max(auc, key=auc.get); report["teacher"][f"fold{fi}"] = dict(train_auroc=auc, picked=pick)
    for n in np.unique(chunk_pt[np.isin(seg_pt, test)]):
        m = chunk_pt == n; e = E[n][tok[m, 2], tok[m, 1], tok[m, 0]]
        s_probe[m] = full.predict_proba(e)[:, 1]; s_p128[m] = np.mean([c.predict_proba(e)[:, 1] for c in small], axis=0)
    tm = np.isin(seg_pt, test); t[tm] = W[pick][tm]

best_train = max(max(v["train_auroc"].values()) for v in report["teacher"].values())
worst_train = min(max(v["train_auroc"].values()) for v in report["teacher"].values())
print("teacher training-segment AUROC per fold:", {k: {d: round(a, 3) for d, a in v["train_auroc"].items()} for k, v in report["teacher"].items()})
if best_train < 0.58:
    print("TEACHER GATE: noise (AUROC < 0.58) -> pre-registered stop"); report["gate"] = "noise"
elif worst_train >= 0.80:
    print("TEACHER GATE: not weak (AUROC >= 0.80) -> test uninformative for the weak-teacher case"); report["gate"] = "strong"
else:
    report["gate"] = "weak (informative)"

arms = {"T": t, "T*S_proto": t * s_proto, "T*S_probe128": t * s_p128, "T*S_probe": t * s_probe, "S_probe": s_probe}
report["pooled"] = {k: dict(auprc=float(AP(y, v)), auroc=float(AUC(y, v))) for k, v in arms.items()}
print("pooled:", {k: {m: round(x, 3) for m, x in d.items()} for k, d in report["pooled"].items()})
per = {k: {} for k in arms}
for s in segs:
    m = seg_pt == s
    for k, v in arms.items():
        per[k][s] = dict(auprc=AP(y[m], v[m]), auroc=AUC(y[m], v[m]), n=int(m.sum()), ink_rate=float(y[m].mean()))
report["per_segment"] = per
print("per-segment n / ink rate:", {s: (per["T"][s]["n"], round(per["T"][s]["ink_rate"], 3)) for s in segs})
tests = [("(a)", "T*S_proto", "T"), ("(b)", "T*S_probe128", "T*S_proto"), ("(c1)", "T*S_probe", "T*S_proto"),
         ("(c2)", "T*S_probe", "S_probe"), ("extra", "T*S_probe", "T"), ("extra", "T*S_probe128", "T")]
report["tests"] = {}
for tag, a, b in tests:
    for metric in ("auprc", "auroc"):
        d = np.array([per[a][s][metric] - per[b][s][metric] for s in segs])
        tt = stats.ttest_1samp(d, 0); w = stats.wilcoxon(d); pos = int((d > 0).sum())
        sign_p = stats.binomtest(pos, len(d), 0.5).pvalue
        report["tests"][f"{tag} {a} - {b} {metric}"] = dict(per_segment=d.round(4).tolist(), mean=float(d.mean()),
                                                            t_p=float(tt.pvalue), wilcoxon_p=float(w.pvalue), positive=f"{pos}/8", sign_p=float(sign_p))
        print(f"{tag:5s} {a:>13s} - {b:<12s} {metric}: mean {d.mean():+.3f}  per-seg {np.round(d, 3).tolist()}  "
              f"t p={tt.pvalue:.3f} Wilcoxon p={w.pvalue:.3f} sign {pos}/8 (p={sign_p:.3f})")
json.dump(report, open("weakteacher_results.json", "w"), indent=1, default=float)
