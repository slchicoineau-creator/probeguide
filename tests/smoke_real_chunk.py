"""Smoke test on real data: run ProbeGuidedLabelGenerator (and villa's DinoGuidedLabelGenerator for comparison) on one
256^3 chunk of PHerc. Paris 4 and report label fractions. Requires the HF checkpoints in ./assets (see README)."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("VILLA_SRC", "../villa/vesuvius/src"))
from probeguide.generator import ProbeGuidedLabelGenerator
from probeguide.zio import RemoteArray, CT_2UM
from vesuvius.ink_detection.training.dynamic_labels import DinoGuidedLabelGenerator
A = os.environ.get("PROBEGUIDE_ASSETS", "assets")
x0, y0, z0 = [int(v) for v in sys.argv[1:4]] if len(sys.argv) > 3 else (15000, 16000, 47000)
v = RemoteArray(CT_2UM, 0).read(z0, z0 + 256, y0, y0 + 256, x0, x0 + 256).astype(np.float32)
lo, hi = np.percentile(v, [1, 99]); v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
img = torch.tensor(v)[None, None]
BB = next((f"{A}/{n}" for n in ("dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt", "dinovol_teacher_backbone.pt")
           if os.path.exists(f"{A}/{n}")), f"{A}/dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt")
# team config (dino_guided_v1/v2.json): stride 128; minibatch 2 fits a 16 GB GPU (the team uses 16 on larger cards)
cfg = dict(unet_ckpt=f"{A}/ckpt_064000.pth", dino_ckpt=BB, dino_stride=128, dino_minibatch=2)
t = time.time()
pg = ProbeGuidedLabelGenerator.from_mapping(dict(cfg, probe_weights=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "probe_surface.npz")))
lab_p = pg.generate(img); s_p = pg._dino_similarity(img.to("cuda", torch.bfloat16))
print(f"ProbeGuide: label shape {tuple(lab_p.shape)}, label fraction {lab_p.mean().item():.4f}, S range "
      f"[{s_p.min().item():.3f}, {s_p.max().item():.3f}]  ({time.time()-t:.0f}s)")
del pg; torch.cuda.empty_cache()
dg = DinoGuidedLabelGenerator.from_mapping(dict(cfg, ref_embedding=f"{A}/avg_ref_embedding.npy"))
lab_d = dg.generate(img); s_d = dg._dino_similarity(img.to("cuda", torch.bfloat16))
print(f"incumbent:  label shape {tuple(lab_d.shape)}, label fraction {lab_d.mean().item():.4f}, S range "
      f"[{s_d.min().item():.3f}, {s_d.max().item():.3f}]  (min-max forces [0,1])")
inter = (lab_p.bool() & lab_d.bool()).sum().item(); union = (lab_p.bool() | lab_d.bool()).sum().item()
print(f"positive-voxel overlap of the two label volumes: Dice {2 * inter / max(lab_p.sum().item() + lab_d.sum().item(), 1):.3f}, "
      f"IoU {inter / max(union, 1):.3f}")
