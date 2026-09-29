"""Judge gate check (a): Section 3a token-level, pooled vs per-segment vs WITHIN-CHUNK. Replicates eval_variants.py.
Runs in $PROBEGUIDE_DATA. Args: [--dense]. Within-chunk rule fixed before looking: a chunk counts if it has >= 10 tokens of
each class; per-segment value = mean over its qualifying chunks; paired tests n=8."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA); written by the pre-build judge gate
import glob, sys, numpy as np
from scipy import stats
from skimage.filters import threshold_otsu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

DENSE = "--dense" in sys.argv; MINC = 10
rng = np.random.default_rng(0)
ref = np.load(f"{_env.ASSETS}/avg_ref_embedding.npy").astype(np.float32); ref /= np.linalg.norm(ref)
C = []
for n, f in enumerate(sorted(glob.glob("emb/chunk_*.npz"))):
    d = np.load(f); sup = d["sup"] > 0; lab = d["ink"]
    if DENSE:
        dd = np.load(f.replace("chunk_", "dense_")); sup = dd["n_sup"] > 0; lab = (dd["ink_frac"] > 0.5).astype(np.uint8)
    E = d["emb"].astype(np.float32); E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9
    cos = E @ ref; inc = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-6)
    C.append(dict(n=n, seg=str(d["seg"]), kind=str(d["kind"]), y=lab[sup].astype(int), emb=E[sup], inc128=inc[sup],
                  otsu128=cos[sup] - threshold_otsu(cos), raw_cos=cos[sup], all_emb_sub=None))
    # for probe min-max over the whole chunk we need the probe on all chunk tokens; keep a strided subsample + sup tokens
    C[-1]["E_all"] = E.reshape(-1, E.shape[-1])
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
            pa = m.decision_function(c["E_all"]); pa = 1 / (1 + np.exp(-pa))       # probe over the WHOLE chunk
            c["probe_mm"] = (c["probe"] - pa.min()) / max(pa.max() - pa.min(), 1e-6)  # recipe-style chunk min-max
for c in C:
    del c["E_all"]
    c["kindflag"] = np.full(len(c["y"]), 1.0 if c["kind"] == "rich" else 0.0)

arms = ["inc128", "otsu128", "raw_cos", "probe_128", "probe", "probe_mm", "kindflag"]
y = np.concatenate([c["y"] for c in C]); sg = np.concatenate([np.full(len(c["y"]), c["seg"]) for c in C])
S = {k: np.concatenate([c[k] for c in C]) for k in arms}


def per_seg(k, metric):
    return np.array([metric(y[sg == s], S[k][sg == s]) for s in segs])


def within(k, metric):
    out, nch = [], []
    for s in segs:
        v = [metric(c["y"], c[k]) for c in C if c["seg"] == s and (c["y"] == 1).sum() >= MINC and (c["y"] == 0).sum() >= MINC]
        out.append(np.mean(v)); nch.append(len(v))
    return np.array(out), nch


def test(d):
    w = stats.wilcoxon(d); pos = int((d > 0).sum())
    return f"mean {d.mean():+.3f} per-seg {np.round(d, 3).tolist()} t p={stats.ttest_1samp(d, 0).pvalue:.3f} W p={w.pvalue:.3f} {pos}/8 sign p={stats.binomtest(pos, 8, .5).pvalue:.3f}"


print("labels", "dense" if DENSE else "sparse", "tokens", len(y), "ink rate", round(y.mean(), 3))
print(f"{'arm':10s} pooledPR pooledROC | perseg PR  ROC | within PR  ROC")
for k in arms:
    wp, nch = within(k, AP); wr, _ = within(k, AUC)
    print(f"{k:10s} {AP(y, S[k]):.3f}   {AUC(y, S[k]):.3f}   | {per_seg(k, AP).mean():.3f} {per_seg(k, AUC).mean():.3f} | {wp.mean():.3f} {wr.mean():.3f}")
print("qualifying chunks per segment (>=%d each class):" % MINC, nch, "of", [sum(c['seg'] == s for c in C) for s in segs])
for a, b in [("probe", "inc128"), ("probe_128", "inc128"), ("probe_mm", "inc128"), ("probe", "probe_mm")]:
    for mn, met in (("AUPRC", AP), ("AUROC", AUC)):
        print(f"per-seg   {a}-{b} {mn}: {test(per_seg(a, met) - per_seg(b, met))}")
        print(f"within    {a}-{b} {mn}: {test(within(a, met)[0] - within(b, met)[0])}")
