"""Build the held-out 3D token-label set for ProbeGuide from the public Paris 4 ink labels (version 20260918).

For each labeled segment:
  * 2D ink labels + supervision mask at canvas level 2 (9.6 um / px);
  * tifxyz mesh (2.4 um frame, grid scale 0.05 => 1 vertex per 20 level-0 px => 5 level-2 px): bilinear map of each
    canvas pixel to base-voxel xyz;
  * choose 256^3 evaluation chunks centred on supervised surface points (mix of ink-rich and ink-free chunks),
    excluding the 8 `extra_patches` centres (+-1024 jitter) the ink_3d model oversampled during training;
  * store, per chunk, the supervised label points (xyz, ink 0/1).
Output: points_<segment>.npz + chunks.json
"""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import json, sys, io, urllib.request
import numpy as np
import tifffile
import zarr
from zarr.storage import FsspecStore

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments"
SEGS = ["20230702185753", "20230929220926", "20231007101619", "20231012184424",
        "20231016151002", "20231031143852", "20231106155351", "20231210121321"]
EXTRA = np.array([[22167, 8793, 37628], [19735, 19651, 35344], [10649, 19605, 70688], [12261, 16221, 41802],
                  [8270, 10039, 41652], [8006, 8812, 41652], [19310, 10476, 41652], [27405, 17662, 41778]], float)
LEVEL, CH = 2, 256
rng = np.random.default_rng(0)


def fetch(url):
    import time
    for attempt in range(8):
        try:
            return urllib.request.urlopen(url, timeout=600).read()
        except Exception as e:
            print("retry", attempt, e, flush=True); time.sleep(10 * (attempt + 1))
    raise RuntimeError(url)


def tif(url):
    import os, hashlib, time
    os.makedirs("tifcache", exist_ok=True)
    f = "tifcache/" + url.split("/segments/")[1].replace("/", "_")
    if not os.path.exists(f):
        for attempt in range(6):
            try:
                data = fetch(url); break
            except Exception as e:
                print("retry", attempt, e, flush=True); time.sleep(10 * (attempt + 1))
        open(f, "wb").write(data)
    return tifffile.imread(f)


def label_level(seg, name):
    g = zarr.open_group(FsspecStore.from_url(f"{B}/{seg}/ink-labels/2.4um-volume-20260411134726/20260918/{name}.zarr",
                                             read_only=True), mode="r")
    return g[str(LEVEL)][:]


def bilinear(grid, r, c):
    r0 = np.clip(np.floor(r).astype(int), 0, grid.shape[0] - 2); c0 = np.clip(np.floor(c).astype(int), 0, grid.shape[1] - 2)
    fr, fc = r - r0, c - c0
    g00, g01, g10, g11 = grid[r0, c0], grid[r0, c0 + 1], grid[r0 + 1, c0], grid[r0 + 1, c0 + 1]
    ok = (g00 > 0) & (g01 > 0) & (g10 > 0) & (g11 > 0)
    v = g00 * (1 - fr) * (1 - fc) + g01 * (1 - fr) * fc + g10 * fr * (1 - fc) + g11 * fr * fc
    return v, ok


def main():
    chunks = []
    for seg in SEGS:
        mesh = f"{B}/{seg}/mesh/{seg}-on-20260411134726-2.4um.tifxyz"
        meta = json.loads(fetch(f"{mesh}/meta.json"))
        X, Y, Z = (tif(f"{mesh}/{a}.tif").astype(np.float32) for a in "xyz")
        ink = label_level(seg, "inklabels"); sup = label_level(seg, "supervision")
        s = meta["scale"][0] * (2 ** LEVEL)          # canvas-level px -> mesh grid index
        rr, cc = np.nonzero(sup > 0)
        if len(rr) > 400000:
            k = rng.choice(len(rr), 400000, replace=False); rr, cc = rr[k], cc[k]
        x, okx = bilinear(X, rr * s, cc * s); y, oky = bilinear(Y, rr * s, cc * s); z, okz = bilinear(Z, rr * s, cc * s)
        ok = okx & oky & okz
        P = np.stack([x, y, z], 1)[ok]; lab = (ink[rr, cc] > 0)[ok].astype(np.uint8)
        far = np.min(np.linalg.norm(P[:, None, :] - EXTRA[None], axis=2), axis=1) > 1024 + CH
        P, lab = P[far], lab[far]
        np.savez_compressed(f"points_{seg}.npz", xyz=P.astype(np.float32), ink=lab)
        print(seg, "mesh", X.shape, "labels", ink.shape, "supervised pts", len(P), "ink frac", lab.mean().round(3), flush=True)
        # chunk centres: 4 ink-rich + 2 ink-poor per segment (by ink fraction of points within +-96 vx)
        cand = P[rng.choice(len(P), min(20000, len(P)), replace=False)]
        picks = {"rich": [], "poor": []}
        for c in cand:
            near = np.all(np.abs(P - c) < 96, axis=1)
            if near.sum() < 50:
                continue
            f = lab[near].mean()
            o = (np.round(c) - CH // 2).astype(int)
            if any(np.all(np.abs(np.array(q["origin_xyz"]) - o) < CH) for q in chunks):
                continue
            kind = "rich" if f > 0.15 else ("poor" if f < 0.02 else None)
            if kind and len(picks[kind]) < (5 if kind == "rich" else 2):
                picks[kind].append(o); chunks.append(dict(segment=seg, kind=kind, origin_xyz=o.tolist(),
                                                         n_points=int(near.sum()), ink_frac=float(f)))
            if len(picks["rich"]) == 5 and len(picks["poor"]) == 2:
                break
        print("   chunks", {k: len(v) for k, v in picks.items()}, flush=True)
    json.dump(chunks, open("chunks.json", "w"), indent=1)
    print("total chunks", len(chunks))


if __name__ == "__main__":
    main()
