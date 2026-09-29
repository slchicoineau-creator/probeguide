"""Within-chunk version of README 3b (U-Net x S, fulllabel128_rows.npy). Chunk counts if >= 10 points of each class."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA); written by the pre-build judge gate
import sys, numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
rows = list(np.load("fulllabel128_rows.npy", allow_pickle=True)); segs = sorted({r["seg"] for r in rows})
for pre in ("v1_", "v2_"):
    for a, b in (("probe", "inc"), ("unet", "inc"), ("unet", "probe")):
        for mn, met in (("AUPRC", AP), ("AUROC", AUC)):
            ps, wc, nch = [], [], []
            for s in segs:
                rs = [r for r in rows if r["seg"] == s]; yy = np.concatenate([r["y"] for r in rs])
                ps.append(met(yy, np.concatenate([r[pre + a] for r in rs])) - met(yy, np.concatenate([r[pre + b] for r in rs])))
                v = [met(r["y"], r[pre + a]) - met(r["y"], r[pre + b]) for r in rs if (r["y"] == 1).sum() >= 10 and (r["y"] == 0).sum() >= 10]
                wc.append(np.mean(v)); nch.append(len(v))
            for tag, d in (("per-seg", np.array(ps)), ("within ", np.array(wc))):
                pos = int((d > 0).sum())
                print(f"{pre}{a}-{b} {mn} {tag}: mean {d.mean():+.3f} t p={stats.ttest_1samp(d, 0).pvalue:.3f} W p={stats.wilcoxon(d).pvalue:.3f} {pos}/8")
print("qualifying chunks per segment:", nch)
