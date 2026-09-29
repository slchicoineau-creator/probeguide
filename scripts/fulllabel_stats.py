"""Every full-label (UNet x S) statistic quoted in the README, from fulllabel.py's per-point rows:
pooled AUPRC/AUROC per arm, paired segment cluster-bootstrap CIs, and the stricter per-SEGMENT paired tests
(n = 8 segments: t-test, Wilcoxon, sign count) — with only 8 clusters the bootstrap is anti-conservative."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import json, sys, numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
# usage: python fulllabel_stats.py [rows.npy] [prefix]   e.g.  fulllabel128_rows.npy v2_
src = sys.argv[1] if len(sys.argv) > 1 else "fulllabel_rows.npy"; pre = sys.argv[2] if len(sys.argv) > 2 else ""
rows = list(np.load(src, allow_pickle=True)); rng = np.random.default_rng(0)
segs = sorted({r["seg"] for r in rows}); arms = [k for k in ["inc", "otsu", "probe", "unet"] if pre + k in rows[0]]
rows = [dict(r, **{k: r[pre + k] for k in arms}) for r in rows]
cat = lambda k, rs: np.concatenate([r[k] for r in rs])
y = cat("y", rows); out = {"pooled": {k: dict(auprc=AP(y, cat(k, rows)), auroc=AUC(y, cat(k, rows))) for k in arms}}
print("pooled:", {k: {m: round(v, 3) for m, v in d.items()} for k, d in out["pooled"].items()})
pairs = [("probe", "inc"), ("unet", "inc"), ("unet", "probe")] + ([("probe", "otsu"), ("otsu", "inc")] if "otsu" in arms else [])
for a, b in pairs:
    for name, fn in [("AUPRC", AP), ("AUROC", AUC)]:
        boot = []
        for _ in range(2000):
            rs = [r for s in rng.choice(segs, len(segs)) for r in rows if r["seg"] == s]; yy = cat("y", rs)
            boot.append(fn(yy, cat(a, rs)) - fn(yy, cat(b, rs)))
        per = []
        for s in segs:
            rs = [r for r in rows if r["seg"] == s]; yy = cat("y", rs)
            if 0 < yy.mean() < 1: per.append(fn(yy, cat(a, rs)) - fn(yy, cat(b, rs)))
        per = np.array(per); t = stats.ttest_1samp(per, 0); w = stats.wilcoxon(per)
        ci_t = stats.t.interval(0.95, len(per) - 1, loc=per.mean(), scale=stats.sem(per))
        key = f"{a}-{b} {name}"
        out[key] = dict(bootstrap_mean=float(np.mean(boot)), bootstrap_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        per_segment=per.round(4).tolist(), seg_mean=float(per.mean()), seg_t_ci=[float(ci_t[0]), float(ci_t[1])],
                        t_p=float(t.pvalue), wilcoxon_p=float(w.pvalue), positive=f"{int((per > 0).sum())}/{len(per)}")
        print(f"{key:18s} bootstrap {np.mean(boot):+.3f} [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}] | "
              f"per-segment mean {per.mean():+.3f} t-CI [{ci_t[0]:+.3f}, {ci_t[1]:+.3f}] p={t.pvalue:.3f} Wilcoxon p={w.pvalue:.3f} "
              f"positive {int((per > 0).sum())}/{len(per)}")

# the binary label the recipe actually emits: score > 0.5 (shipped threshold), pooled and per-segment F1 differences
def prf(yy, s):
    p = s > 0.5; tp = (p & (yy == 1)).sum(); P = tp / max(p.sum(), 1); R = tp / max((yy == 1).sum(), 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)
out["at_0.5"] = {k: dict(zip(["precision", "recall", "f1"], map(float, prf(y, cat(k, rows))))) for k in arms}
print("at shipped threshold 0.5:", {k: {m: round(v, 3) for m, v in d.items()} for k, d in out["at_0.5"].items()})
for a, b in [pr for pr in pairs if pr[1] == "inc"]:
    per = np.array([prf(cat("y", rs), cat(a, rs))[2] - prf(cat("y", rs), cat(b, rs))[2]
                    for s in segs for rs in [[r for r in rows if r["seg"] == s]]])
    t = stats.ttest_1samp(per, 0)
    out[f"{a}-{b} F1@0.5"] = dict(per_segment=per.round(4).tolist(), seg_mean=float(per.mean()), t_p=float(t.pvalue))
    print(f"{a}-{b} F1@0.5 per-segment mean {per.mean():+.3f} p={t.pvalue:.3f} positive {int((per > 0).sum())}/{len(per)}")
json.dump(out, open(f"fulllabel_stats_{src.replace('.npy', '')}{('_' + pre.strip('_')) if pre else ''}.json", "w"), indent=1, default=float)
