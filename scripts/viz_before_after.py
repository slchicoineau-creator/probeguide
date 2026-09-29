import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import os
os.makedirs("figs", exist_ok=True)
import glob, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from probeguide.zio import RemoteArray, CT_2UM
ref = np.load(f"{_env.ASSETS}/avg_ref_embedding.npy").astype(np.float32); ref /= np.linalg.norm(ref)
files = sorted(glob.glob("emb/chunk_*.npz")); D = [np.load(f) for f in files]
segs = sorted({str(d["seg"]) for d in D}); folds = [segs[i::4] for i in range(4)]
def probe_for(seg):
    test = [f for f in folds if seg in f][0]; X, y = [], []
    for d in D:
        if str(d["seg"]) in test: continue
        s = d["sup"] > 0; e = d["emb"][s].astype(np.float32); X.append(e / np.linalg.norm(e, axis=1, keepdims=True)); y.append(d["ink"][s])
    return LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(np.concatenate(X), np.concatenate(y))
picks = [int(a) for a in sys.argv[1:]] or [2, 1]
fig, axs = plt.subplots(len(picks), 4, figsize=(20, 5.2 * len(picks)))
for row, n in zip(axs, picks):
    d = D[n]; seg = str(d["seg"]); x0, y0, z0 = d["origin_xyz"]; m = probe_for(seg)
    e = d["emb"].reshape(-1, 864).astype(np.float32); e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
    pg = m.predict_proba(e)[:, 1].reshape(32, 32, 32)
    cos = (e @ ref).reshape(32, 32, 32); inc = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-6)  # team stride 128
    P = np.load(f"points_{seg}.npz"); rel = P["xyz"] - np.array([x0, y0, z0]); ok = np.all((rel >= 0) & (rel < 256), 1)
    rel, ink = rel[ok], P["ink"][ok]; kz = np.bincount((rel[:, 2] // 8).astype(int), minlength=32).argmax(); zc = kz * 8 + 4
    ct = RemoteArray(CT_2UM, 0).read(z0 + zc, z0 + zc + 1, y0, y0 + 256, x0, x0 + 256)[0]; sl = np.abs(rel[:, 2] - zc) < 8
    row[0].imshow(ct, cmap="gray"); row[0].set_title(f"{seg} ({d['kind']}) z={z0+zc}: CT")
    row[1].imshow(ct, cmap="gray"); row[1].scatter(rel[sl, 0], rel[sl, 1], c=np.where(ink[sl] > 0, "red", "cyan"), s=5)
    row[1].set_title("held-out labels: red ink, cyan no ink")
    for a, grid, t in [(row[2], inc[kz], "incumbent S (stride 128)"), (row[3], pg[kz], "ProbeGuide S (held-out probe)")]:
        a.imshow(ct, cmap="gray"); im = a.imshow(np.kron(grid, np.ones((8, 8))), alpha=.5, cmap="magma", vmin=0, vmax=1)
        a.contour(np.kron(grid, np.ones((8, 8))), levels=[0.5], colors="lime", linewidths=.8); a.set_title(t + ", green = 0.5")
    for a in row: a.axis("off")
plt.tight_layout(); plt.savefig("figs/before_after_maps.png", dpi=70); print("ok")
