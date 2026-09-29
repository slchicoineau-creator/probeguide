"""Embed every evaluation chunk with the frozen Dinovol teacher and reproduce the incumbent similarity exactly.

Per 256^3 chunk (L0 of PHercParis4 2.4 um scan, percentile p1/p99 min-max normalisation like `percentile_minmax`):
  * 27 windows of 128^3 at stride 64 (DinoGuidedLabelGenerator defaults: chunk 256, window 128, stride 64, sigma 4);
  * incumbent token score = Gaussian-blended cosine(token, avg_ref_embedding) on the 32^3 token grid, then min-max
    normalised over the chunk (exactly `_dino_similarity`, evaluated on the token grid before trilinear upsampling);
  * raw blended cosine (same, without min-max);
  * token embeddings (fp16) of the 8 non-overlapping windows (starts 0/128) -> 32^3 x 864, for the probes;
  * token labels from the 2D->3D mapped label points: ink = any ink point in the token, sup = number of points.
"""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import json, sys, time, os
import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from probeguide.zio import RemoteArray, CT_2UM

ASSETS = _env.ASSETS
ck = torch.load(_env.backbone_path(), map_location="cpu", weights_only=False)
state = {k.removeprefix("backbone."): v for k, v in ck["teacher"].items() if k.startswith("backbone.")}
bb = build_dinovol_2_backbone(ck["config"]["model"]); bb.load_pretrained_weights(state)
bb = bb.cuda().eval().to(torch.bfloat16)
ref = torch.tensor(np.load(f"{ASSETS}/avg_ref_embedding.npy"), dtype=torch.float32, device="cuda")
ref = ref / ref.norm()

W, P, S = 128, 8, 64
T = W // P                                   # 16 tokens per window axis
starts = [0, 64, 128]
ax = torch.arange(T, dtype=torch.float32) - (T - 1) / 2
g1 = torch.exp(-0.5 * (ax / 4.0) ** 2); g1 = g1 / g1.max()
weight = (g1[:, None, None] * g1[None, :, None] * g1[None, None, :]).cuda()

ct = RemoteArray(CT_2UM, 0)
chunks = json.load(open("chunks.json"))
os.makedirs("emb", exist_ok=True)
pts = {}
for n, c in enumerate(chunks):
    out = f"emb/chunk_{n:03d}.npz"
    if os.path.exists(out):
        continue
    t0 = time.time()
    x0, y0, z0 = c["origin_xyz"]
    vol = ct.read(z0, z0 + 256, y0, y0 + 256, x0, x0 + 256).astype(np.float32)
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    img = torch.tensor(vol, device="cuda", dtype=torch.bfloat16)[None, None]
    acc = torch.zeros((32, 32, 32), device="cuda"); wacc = torch.zeros_like(acc)
    emb = torch.zeros((32, 32, 32, 864), dtype=torch.float16)
    with torch.inference_mode():
        coords = [(a, b, d) for a in starts for b in starts for d in starts]
        for k in range(0, len(coords), 4):
            batch = coords[k:k + 4]
            win = torch.cat([img[:, :, a:a + W, b:b + W, d:d + W] for a, b, d in batch])
            tok = bb.forward_features(win)["x_norm_patchtokens"].float()          # (n, 4096, 864)
            sim = (F.normalize(tok, dim=-1) @ ref).reshape(len(batch), T, T, T)
            for j, (a, b, d) in enumerate(batch):
                sl = (slice(a // P, a // P + T), slice(b // P, b // P + T), slice(d // P, d // P + T))
                acc[sl] += sim[j] * weight; wacc[sl] += weight
                if a % 128 == 0 and b % 128 == 0 and d % 128 == 0:
                    emb[sl] = tok[j].reshape(T, T, T, 864).half().cpu()
    raw = (acc / wacc.clamp_min(1e-6)).cpu().numpy()
    inc = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-6)
    # token labels
    seg = c["segment"]
    if seg not in pts:
        d = np.load(f"points_{seg}.npz"); pts[seg] = (d["xyz"], d["ink"])
    xyz, ink = pts[seg]
    rel = xyz - np.array([x0, y0, z0], np.float32)
    m = np.all((rel >= 0) & (rel < 256), axis=1)
    ti = (rel[m] // 8).astype(int)                                   # x, y, z token index
    lab_ink = np.zeros((32, 32, 32), np.uint8); lab_sup = np.zeros((32, 32, 32), np.int32)
    np.add.at(lab_sup, (ti[:, 2], ti[:, 1], ti[:, 0]), 1)
    np.maximum.at(lab_ink, (ti[:, 2], ti[:, 1], ti[:, 0]), ink[m])
    np.savez_compressed(out, emb=emb.numpy(), raw_cos=raw.astype(np.float32), incumbent=inc.astype(np.float32),
                        ink=lab_ink, sup=lab_sup, ct_mean=np.float32(vol.mean()), seg=seg, kind=c["kind"],
                        origin_xyz=np.array(c["origin_xyz"]))
    print(n, seg, c["kind"], "sup tokens", int((lab_sup > 0).sum()), "ink tokens", int(lab_ink.sum()),
          f"{time.time() - t0:.0f}s", flush=True)
