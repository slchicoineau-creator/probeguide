"""Full-label test at the TEAM'S ACTUAL settings (judge fixes B1/B2, non-blocking #6/#8):
  * stride 128 (team configs dino_guided_v1/v2.json): 8 non-overlapping windows -> token score = cos(token, ref);
    incumbent S = min-max over the 256^3 chunk (applied after trilinear upsampling, exactly as `_dino_similarity`);
  * arms: UNet x incumbent S, UNet x Otsu-binarised cosine (team's newer DINO-guidance binarisation), UNet x ProbeGuide
    (fold probe), UNet alone;
  * v1 rule (no intensity gate) and v2 rule (raw CT > 50 foreground gate, `input_mask_threshold: 50`);
  * scored at held-out label points with a +-3 voxel tolerance (max over 7^3 = the recipe's projection half-thickness 3).
Per-point values are saved (fulllabel128_rows.npy) so statistics can be recomputed on CPU (fulllabel_stats.py)."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, numpy as np, torch, torch.nn.functional as F
from skimage.filters import threshold_otsu
from sklearn.linear_model import LogisticRegression
from vesuvius.ink_detection.training.dynamic_labels import load_frozen_ink_model
from vesuvius.ink_detection.inference.inference_runtime import TargetModel
from probeguide.zio import RemoteArray, CT_2UM

unet = TargetModel(load_frozen_ink_model(f"{_env.ASSETS}/ckpt_064000.pth", device="cuda", dtype=torch.bfloat16))
ref = np.load(f"{_env.ASSETS}/avg_ref_embedding.npy").astype(np.float32); ref /= np.linalg.norm(ref)
ct = RemoteArray(CT_2UM, 0)
D = [np.load(f) for f in sorted(glob.glob("emb/chunk_*.npz"))]
segs = sorted({str(d["seg"]) for d in D}); folds = [segs[i::4] for i in range(4)]
pts = {s: np.load(f"points_{s}.npz") for s in segs}
probe = {}
for fi, test in enumerate(folds):
    X, y = [], []
    for d in D:
        if str(d["seg"]) in test: continue
        s = d["sup"] > 0; e = d["emb"][s].astype(np.float32); X.append(e / np.linalg.norm(e, axis=1, keepdims=True)); y.append(d["ink"][s])
    probe[fi] = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(np.concatenate(X), np.concatenate(y))

up = lambda g: F.interpolate(torch.tensor(g, dtype=torch.float32, device="cuda")[None, None], size=(256,) * 3,
                             mode="trilinear", align_corners=False)[0, 0]
mp = lambda v: F.max_pool3d(v[None, None], 7, 1, 3)[0, 0]
rows = []
for n, d in enumerate(D):
    seg = str(d["seg"]); fi = [i for i, f in enumerate(folds) if seg in f][0]; x0, y0, z0 = d["origin_xyz"]
    raw = ct.read(z0, z0 + 256, y0, y0 + 256, x0, x0 + 256).astype(np.float32)
    lo, hi = np.percentile(raw, [1, 99]); v = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)
    E = d["emb"].astype(np.float32); E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9
    cos = E @ ref; pg = probe[fi].predict_proba(E.reshape(-1, 864))[:, 1].reshape(32, 32, 32)
    with torch.inference_mode():
        prob = unet(torch.tensor(v, device="cuda", dtype=torch.bfloat16)[None, None]).float().sigmoid()[0, 0]
        gate = torch.tensor(raw > 50, device="cuda").float()
        c = up(cos); s_inc = (c - c.min()) / (c.max() - c.min()).clamp_min(1e-6)
        s_otsu = (c > float(threshold_otsu(cos))).float(); s_pr = up(pg)
        P = pts[seg]; rel = P["xyz"] - np.array([x0, y0, z0], np.float32)
        m = np.all((rel >= 3) & (rel < 253), axis=1); ix = rel[m].astype(int)
        zz, yy, xx = (torch.tensor(a, device="cuda") for a in (ix[:, 2], ix[:, 1], ix[:, 0]))
        row = dict(seg=seg, fold=fi, kind=str(d["kind"]), y=P["ink"][m])
        for rule, g in [("v1", 1.0), ("v2", gate)]:
            pr = prob * g
            for arm, S in [("inc", s_inc), ("otsu", s_otsu), ("probe", s_pr), ("unet", None)]:
                val = pr if S is None else pr * S
                row[f"{rule}_{arm}"] = mp(val)[zz, yy, xx].cpu().numpy()
    rows.append(row); print(n, seg, str(d["kind"]), "pts", int(m.sum()), flush=True)
np.save("fulllabel128_rows.npy", rows, allow_pickle=True)
print("saved fulllabel128_rows.npy")
