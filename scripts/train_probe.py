"""Train the release probes on ALL 8 labeled Paris 4 segments (frozen dinovol teacher tokens, L2-normalised).
  probe_surface.npz  : supervised surface tokens only (default inside the DINO-guided recipe, where the UNet masks air)
  probe_with_air.npz : + low-intensity 'air' tokens as negatives (standalone use)"""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, json, os, numpy as np
os.makedirs("results", exist_ok=True)
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0); X, y, A = [], [], []
for f in sorted(glob.glob("emb/chunk_*.npz")):
    d = np.load(f); s = d["sup"] > 0; it = np.load(f.replace("chunk_", "int_").replace(".npz", ".npy"))[..., 0]
    E = d["emb"].astype(np.float32); E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9
    X.append(E[s]); y.append(d["ink"][s])
    air = np.argwhere((it < np.quantile(it, 0.15)) & ~s); air = air[rng.choice(len(air), min(300, len(air)), replace=False)]
    A.append(E[tuple(air.T)])
X, y, A = np.concatenate(X), np.concatenate(y), np.concatenate(A)
meta = dict(backbone="scrollprize/dinovol_v2_ps8_with_paris4_352500 teacher; x_norm_patchtokens; L2-normalised",
            training_data="PHerc. Paris 4 ink labels 20260918, 8 segments, 56 chunks of 256^3 (see README)")
for name, XX, yy in [("probe_surface", X, y), ("probe_with_air", np.concatenate([X, A]), np.r_[y, np.zeros(len(A))])]:
    m = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(XX, yy)
    thr_f1 = np.nan
    if name == "probe_surface" and os.path.exists("fixed_threshold_results.json"):
        thr_f1 = json.load(open("fixed_threshold_results.json"))["thresholds"]["probe"]
    np.savez(f"results/{name}.npz", coef=m.coef_[0].astype(np.float32), intercept=np.float32(m.intercept_[0]),
             n_tokens=len(yy), threshold_shipped=np.float32(0.5),
             threshold_bestf1_with_ckpt064000=np.float32(thr_f1),
             note="label = UNet x probe > threshold. 0.5 = shipped setting; the best-F1 threshold was chosen on "
                  "training folds with UNet = ink_3d_dino_guided ckpt_064000 and only applies to that UNet.", **meta)
    print(name, "tokens", len(yy))
