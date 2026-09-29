"""Deployable operating points for the full label (UNet x S), from fulllabel.py's per-point scores:
(a) the shipped threshold 0.5 for both arms; (b) one fixed threshold per arm = median of the four train-fold best-F1
thresholds. Reports precision / recall / F1 / FP (overall and on ink-poor chunks) and segment-bootstrap CIs."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import json, numpy as np
from sklearn.metrics import precision_recall_curve
rows = list(np.load("fulllabel_rows.npy", allow_pickle=True)); segs = sorted({r["seg"] for r in rows})
rng = np.random.default_rng(0)
def best_t(y, s):
    p, r, t = precision_recall_curve(y, s); f = 2 * p * r / np.maximum(p + r, 1e-9); return t[np.argmax(f[:-1])]
thr = {k: float(np.median([best_t(np.concatenate([r["y"] for r in rows if r["fold"] != fi]),
                                   np.concatenate([r[k] for r in rows if r["fold"] != fi])) for fi in range(4)]))
       for k in ["inc", "probe"]}
def stats(rs, k, t):
    y = np.concatenate([r["y"] for r in rs]); p = np.concatenate([r[k] for r in rs]) > t
    poor = np.concatenate([np.full(len(r["y"]), r["kind"] == "poor") for r in rs])
    P = (p & (y == 1)).sum() / max(p.sum(), 1); R = (p & (y == 1)).sum() / max((y == 1).sum(), 1)
    return dict(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9), fp=p[y == 0].mean(), fp_poor=p[poor & (y == 0)].mean())
out = {"thresholds": thr}
for name, (ti, tp) in {"shipped_0.5": (0.5, 0.5), "fixed_bestF1": (thr["inc"], thr["probe"])}.items():
    a, b = stats(rows, "inc", ti), stats(rows, "probe", tp); out[name] = dict(incumbent=a, probe=b)
    boot = {m: [] for m in a}
    for _ in range(2000):
        rs = [r for s in rng.choice(segs, len(segs)) for r in rows if r["seg"] == s]
        x, z = stats(rs, "inc", ti), stats(rs, "probe", tp)
        for m in boot: boot[m].append(z[m] - x[m])
    out[name]["diff_CI"] = {m: [float(np.percentile(v, 2.5)), float(np.mean(v)), float(np.percentile(v, 97.5))] for m, v in boot.items()}
    print(name, f"(inc thr {ti:.3f}, probe thr {tp:.3f})")
    for m in a: print(f"   {m:9s} inc {a[m]:.3f}  probe {b[m]:.3f}  diff {out[name]['diff_CI'][m][1]:+.3f} "
                      f"[{out[name]['diff_CI'][m][0]:+.3f}, {out[name]['diff_CI'][m][2]:+.3f}]")
json.dump(out, open("fixed_threshold_results.json", "w"), indent=1, default=float)
