"""Decisive test: the FULL DINO-guided label, label = (ink-UNet probability x S) > t, with S = incumbent similarity
(exact: blended cosine to avg_ref_embedding, min-max per chunk) vs S = ProbeGuide probe probability (linear probe
trained on the OTHER folds' segments). The UNet (`ink_3d_dino_guided` ckpt_064000 — the unpublished stage-1 teacher is
replaced by this public checkpoint; it saw Paris 4 so absolute numbers are inflated for BOTH arms) is shared.
Scored at held-out label points (voxel, max over 3x3x3 to tolerate surface offset)."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, sys, json, numpy as np, torch, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from vesuvius.ink_detection.training.dynamic_labels import load_frozen_ink_model
from vesuvius.ink_detection.inference.inference_runtime import TargetModel
from probeguide.zio import RemoteArray, CT_2UM

unet = TargetModel(load_frozen_ink_model(f"{_env.ASSETS}/ckpt_064000.pth", device="cuda", dtype=torch.bfloat16))
ct = RemoteArray(CT_2UM, 0)
files = sorted(glob.glob("emb/chunk_*.npz"))
D = [np.load(f) for f in files]
segs = sorted({str(d["seg"]) for d in D}); folds = [segs[i::4] for i in range(4)]
pts = {s: np.load(f"points_{s}.npz") for s in segs}

# fold probes (trained on supervised tokens of the other segments)
probe = {}
for fi, test in enumerate(folds):
    X, y = [], []
    for d in D:
        if str(d["seg"]) in test: continue
        sup = d["sup"] > 0; e = d["emb"][sup].astype(np.float32); X.append(e / np.linalg.norm(e, axis=1, keepdims=True)); y.append(d["ink"][sup])
    probe[fi] = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(np.concatenate(X), np.concatenate(y))

def up(grid):  # token grid (32^3) -> 256^3 like the generator (trilinear, align_corners=False)
    return F.interpolate(torch.tensor(grid, dtype=torch.float32, device="cuda")[None, None], size=(256,) * 3,
                         mode="trilinear", align_corners=False)[0, 0]

rows = []
for n, d in enumerate(D):
    seg = str(d["seg"]); fi = [i for i, f in enumerate(folds) if seg in f][0]
    x0, y0, z0 = d["origin_xyz"]
    v = ct.read(z0, z0 + 256, y0, y0 + 256, x0, x0 + 256).astype(np.float32)
    lo, hi = np.percentile(v, [1, 99]); v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    with torch.inference_mode():
        prob = unet(torch.tensor(v, device="cuda", dtype=torch.bfloat16)[None, None]).float().sigmoid()[0, 0]
        e = d["emb"].reshape(-1, 864).astype(np.float32); e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        pgrid = probe[fi].predict_proba(e)[:, 1].reshape(32, 32, 32)
        s_inc = up(d["incumbent"]); s_pr = up(pgrid)
        prod_inc = F.max_pool3d((prob * s_inc)[None, None], 3, 1, 1)[0, 0]
        prod_pr = F.max_pool3d((prob * s_pr)[None, None], 3, 1, 1)[0, 0]
        unet_only = F.max_pool3d(prob[None, None], 3, 1, 1)[0, 0]
    P = pts[seg]; rel = P["xyz"] - np.array([x0, y0, z0], np.float32)
    m = np.all((rel >= 1) & (rel < 255), axis=1); ix = rel[m].astype(int); lab = P["ink"][m]
    zz, yy, xx = map(lambda a: torch.tensor(a, device="cuda"), (ix[:, 2], ix[:, 1], ix[:, 0]))
    rows.append(dict(seg=seg, fold=fi, kind=str(d["kind"]), y=lab, inc=prod_inc[zz, yy, xx].cpu().numpy(),
                     probe=prod_pr[zz, yy, xx].cpu().numpy(), unet=unet_only[zz, yy, xx].cpu().numpy()))
    print(n, seg, str(d["kind"]), "pts", int(m.sum()), flush=True)

np.save("fulllabel_rows.npy", rows, allow_pickle=True)

def f1_at(y, s, t):
    p = s > t; tp = (p & (y == 1)).sum(); return tp / max(p.sum(), 1), tp / max((y == 1).sum(), 1)

print("\nFULL LABEL (UNet x S) at held-out label points")
res = {}
for k in ["inc", "probe", "unet"]:
    per = []
    for fi in range(4):
        tr = [r for r in rows if r["fold"] != fi]; te = [r for r in rows if r["fold"] == fi]
        ytr = np.concatenate([r["y"] for r in tr]); str_ = np.concatenate([r[k] for r in tr])
        pr, rc, th = precision_recall_curve(ytr, str_); f = 2 * pr * rc / np.maximum(pr + rc, 1e-9); t = th[np.argmax(f[:-1])]
        y = np.concatenate([r["y"] for r in te]); s = np.concatenate([r[k] for r in te])
        poor = np.concatenate([np.full(len(r["y"]), r["kind"] == "poor") for r in te])
        p05, r05 = f1_at(y, s, 0.5); pt, rt = f1_at(y, s, t)
        per.append(dict(auprc=average_precision_score(y, s), auroc=roc_auc_score(y, s),
                        f1_05=2 * p05 * r05 / max(p05 + r05, 1e-9), p05=p05, r05=r05,
                        f1_t=2 * pt * rt / max(pt + rt, 1e-9), pt=pt, rt=rt,
                        fp_poor_t=float((s > t)[poor & (y == 0)].mean()) if (poor & (y == 0)).any() else np.nan))
    res[k] = {m: float(np.nanmean([p[m] for p in per])) for m in per[0]}
    print(f"  {k:6s} " + "  ".join(f"{m}={v:.3f}" for m, v in res[k].items()))
json.dump(res, open("fulllabel_results.json", "w"), indent=1)
