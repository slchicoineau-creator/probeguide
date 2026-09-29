"""Dense token labels (judge non-blocking #5): the original token labels came from a 400k-pixel subsample (most tokens
held ONE label point). Here every supervised canvas pixel (level 2) is mapped to 3D and each 8^3 token gets
  n_sup (number of supervised pixels), ink_frac (fraction that are ink) -> label = ink_frac > 0.5 (majority).
Output: emb/dense_XXX.npz next to each chunk (same chunk order). CPU only."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, json, numpy as np
import prepare as P  # reuses the cached mesh reader, label reader and bilinear map (no chunk re-selection)

chunks = json.load(open("chunks.json"))
by_seg = {}
for n, c in enumerate(chunks):
    by_seg.setdefault(c["segment"], []).append((n, np.array(c["origin_xyz"])))
for seg, items in by_seg.items():
    mesh = f"{P.B}/{seg}/mesh/{seg}-on-20260411134726-2.4um.tifxyz"
    meta = json.loads(P.fetch(f"{mesh}/meta.json"))
    X, Y, Z = (P.tif(f"{mesh}/{a}.tif").astype(np.float32) for a in "xyz")
    ink = P.label_level(seg, "inklabels"); sup = P.label_level(seg, "supervision")
    s = meta["scale"][0] * (2 ** P.LEVEL)
    rr, cc = np.nonzero(sup > 0)
    x, okx = P.bilinear(X, rr * s, cc * s); y, oky = P.bilinear(Y, rr * s, cc * s); z, okz = P.bilinear(Z, rr * s, cc * s)
    ok = okx & oky & okz
    xyz = np.stack([x, y, z], 1)[ok]; lab = (ink[rr, cc] > 0)[ok]
    for n, o in items:
        rel = xyz - o; m = np.all((rel >= 0) & (rel < 256), axis=1)
        ti = (rel[m] // 8).astype(int)
        cnt = np.zeros((32, 32, 32), np.int32); inkc = np.zeros((32, 32, 32), np.int32)
        np.add.at(cnt, (ti[:, 2], ti[:, 1], ti[:, 0]), 1); np.add.at(inkc, (ti[:, 2], ti[:, 1], ti[:, 0]), lab[m].astype(np.int32))
        frac = np.where(cnt > 0, inkc / np.maximum(cnt, 1), 0).astype(np.float32)
        np.savez_compressed(f"emb/dense_{n:03d}.npz", n_sup=cnt, ink_frac=frac)
        print(seg, "chunk", n, "supervised tokens", int((cnt > 0).sum()), "median px/token", int(np.median(cnt[cnt > 0])), flush=True)
