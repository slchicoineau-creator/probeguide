"""Weak-teacher test (pre-registered in the project work log, 2026-09-27): does the DINO gate help a teacher that never
saw PHerc. Paris 4, i.e. the new-scroll case the gate exists for?

Teacher T = HF `scrollprize/PHerc.1667-iteration-0` (ResNet3D-50 + 2D U-Net, trained on 500p2a, 658 and two PHerc0009B
segments; no Paris 4). It runs on each Paris 4 segment's 2.4 um surface volume (central 62 of 109 layers, uint8 clipped
to [0, 200] / 255, as its card describes) in both depth directions, 256^2 windows at stride 128, averaged.
A PROXY: the team's own 60k teacher is a 3D volumetric U-Net and is not public.

Points = every supervised level-2 label pixel whose mesh position falls inside one of the 56 evaluation chunks. For each
we store canvas (row, col), xyz, ink, chunk index and T's probability (forward / reversed layers).
Output: wt_points.npz (analysis on CPU in weakteacher_stats.py).
Model files: $PROBEGUIDE_ASSETS/PHerc.1667-iteration-0/{modeling_inkdetection.py, configuration_inkdetection.py,
model.safetensors}."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import dataclasses, importlib.util, json, os, sys, threading, time, types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file
import prepare as P
from probeguide.zio import RemoteArray, S3

MODEL_DIR = f"{_env.ASSETS}/PHerc.1667-iteration-0"
WIN, STRIDE, DEPTH, NLAYERS, BATCH = 256, 128, 62, 109, 4


def load_teacher():
    """Load the HF remote-code model without `transformers` (two tiny stand-ins for its base classes)."""
    tf = types.ModuleType("transformers"); mo = types.ModuleType("transformers.modeling_outputs")

    class PretrainedConfig:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

        @classmethod
        def register_for_auto_class(cls, *a, **k):
            pass

    class PreTrainedModel(nn.Module):
        def __init__(self, config):
            super().__init__(); self.config = config

        @classmethod
        def register_for_auto_class(cls, *a, **k):
            pass

    @dataclasses.dataclass
    class ModelOutput:
        pass

    tf.PretrainedConfig, tf.PreTrainedModel, mo.ModelOutput = PretrainedConfig, PreTrainedModel, ModelOutput
    sys.modules["transformers"], sys.modules["transformers.modeling_outputs"] = tf, mo
    pkg = types.ModuleType("wt_model"); pkg.__path__ = [MODEL_DIR]; sys.modules["wt_model"] = pkg
    mods = {}
    for name in ("configuration_inkdetection", "modeling_inkdetection"):
        spec = importlib.util.spec_from_file_location(f"wt_model.{name}", f"{MODEL_DIR}/{name}.py")
        mods[name] = importlib.util.module_from_spec(spec); sys.modules[f"wt_model.{name}"] = mods[name]
        spec.loader.exec_module(mods[name])
    cfg = json.load(open(f"{MODEL_DIR}/config.json"))
    conf = mods["configuration_inkdetection"].InkDetectionConfig(
        **{k: cfg[k] for k in ("in_channels", "input_depth", "input_size", "backbone_depth", "backbone_channels",
                               "num_classes", "decoder_upscale") if k in cfg})
    for k in ("in_channels", "backbone_depth", "backbone_channels", "decoder_upscale"):
        setattr(conf, k, cfg[k])
    model = mods["modeling_inkdetection"].InkDetectionModel(conf)
    missing, unexpected = model.load_state_dict(load_file(f"{MODEL_DIR}/model.safetensors"), strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.eval().cuda()


def prefetch(sv, idx):
    """Download one zarr chunk into the RemoteArray disk cache (atomic write, so readers never see partial files)."""
    name = sv.sep.join(map(str, idx)); f = sv.cache_dir / name.replace("/", "_")
    if not f.exists():
        from probeguide.zio import _get
        raw = _get(f"{sv.url}/{name}") or b""
        tmp = f.with_suffix(".tmp%d" % threading.get_ident()); tmp.write_bytes(raw); os.replace(tmp, f)


def main():
    chunks = json.load(open("chunks.json"))
    by_seg = {}
    for n, c in enumerate(chunks):
        by_seg.setdefault(c["segment"], []).append((n, np.array(c["origin_xyz"])))
    global MODEL_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("segments", nargs="*", help="optional segment ids (first timing/alignment check)")
    ap.add_argument("--model", default="PHerc.1667-iteration-0", help="teacher folder under $PROBEGUIDE_ASSETS")
    ap.add_argument("--out", default=None, help="output .npz (default wt_points.npz)")
    args = ap.parse_args(); only = args.segments
    MODEL_DIR = f"{_env.ASSETS}/{args.model}"
    # 1) points + windows for every evaluation chunk (CPU)
    jobs = []
    for seg, items in by_seg.items():
        if only and seg not in only:
            continue
        mesh = f"{P.B}/{seg}/mesh/{seg}-on-20260411134726-2.4um.tifxyz"
        meta = json.loads(P.fetch(f"{mesh}/meta.json"))
        X, Y, Z = (P.tif(f"{mesh}/{a}.tif").astype(np.float32) for a in "xyz")
        ink = P.label_level(seg, "inklabels"); sup = P.label_level(seg, "supervision")
        s = meta["scale"][0] * (2 ** P.LEVEL)
        rr, cc = np.nonzero(sup > 0)
        x, okx = P.bilinear(X, rr * s, cc * s); y, oky = P.bilinear(Y, rr * s, cc * s); z, okz = P.bilinear(Z, rr * s, cc * s)
        ok = okx & oky & okz; rr, cc = rr[ok], cc[ok]
        xyz = np.stack([x, y, z], 1)[ok]; lab = (ink[rr, cc] > 0)
        sv = RemoteArray(f"{S3}/PHercParis4/segments/{seg}/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr", 0)
        assert sv.shape[0] == NLAYERS, sv.shape
        for n, o in items:
            m = np.all((xyz - o >= 0) & (xyz - o < 256), axis=1)
            if not m.any():
                continue
            pr, pc = rr[m] * 4 + 2, cc[m] * 4 + 2            # level-0 canvas position of each level-2 pixel centre
            wins = {(int(r0), int(c0)) for r0, c0 in zip(np.floor((pr - WIN // 2) / STRIDE) * STRIDE,
                                                            np.floor((pc - WIN // 2) / STRIDE) * STRIDE)}
            wins = sorted({(r0 + dr, c0 + dc) for r0, c0 in wins for dr in (0, STRIDE) for dc in (0, STRIDE)})  # 2x overlap
            need = sorted({(0, r // sv.chunks[1], c // sv.chunks[2]) for r0, c0 in wins
                           for r in range(max(r0, 0), r0 + WIN, sv.chunks[1]) for c in range(max(c0, 0), c0 + WIN, sv.chunks[2])
                           if r < sv.shape[1] and c < sv.shape[2]})
            jobs.append(dict(seg=seg, n=n, sv=sv, wins=wins, need=need, rr=rr[m], cc=cc[m], xyz=xyz[m], ink=lab[m], pr=pr, pc=pc))
        print(seg, "planned", sum(len(j["wins"]) for j in jobs if j["seg"] == seg), "windows", flush=True)
    # 2) download everything in the background, 32 connections, in processing order
    pool = ThreadPoolExecutor(32)
    for j in jobs:
        j["futs"] = [pool.submit(prefetch, j["sv"], ix) for ix in j["need"]]
    # 3) GPU: batches of windows (both depth directions per window)
    model = load_teacher(); t0 = time.time(); l0 = (NLAYERS - DEPTH) // 2
    out = {k: [] for k in ("seg", "chunk", "rr", "cc", "xyz", "ink", "t_fwd", "t_rev")}
    for j in jobs:
        for f in j["futs"]:
            f.result()
        sv, wins = j["sv"], j["wins"]
        R0, C0 = min(w[0] for w in wins), min(w[1] for w in wins)
        H, W = max(w[0] for w in wins) - R0 + WIN, max(w[1] for w in wins) - C0 + WIN
        acc = np.zeros((2, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        for b in range(0, len(wins), BATCH):
            batch = wins[b:b + BATCH]; vols = []
            for r0, c0 in batch:
                vol = sv.read(0, NLAYERS, r0, r0 + WIN, c0, c0 + WIN)   # out-of-canvas -> fill value 0
                v = np.clip(vol[l0:l0 + DEPTH].astype(np.float32), 0, 200) / 255.0
                vols += [v, v[::-1].copy()]
            t = torch.tensor(np.stack(vols), device="cuda")[:, None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(t).logits.float())
            p = torch.nn.functional.interpolate(p, scale_factor=4, mode="bilinear")[:, 0].cpu().numpy()
            for k, (r0, c0) in enumerate(batch):
                acc[:, r0 - R0:r0 - R0 + WIN, c0 - C0:c0 - C0 + WIN] += p[2 * k:2 * k + 2]
                cnt[r0 - R0:r0 - R0 + WIN, c0 - C0:c0 - C0 + WIN] += 1
        prob = acc / np.maximum(cnt, 1); k = len(j["rr"])
        out["seg"].append(np.full(k, j["seg"])); out["chunk"].append(np.full(k, j["n"]))
        for key in ("rr", "cc", "xyz", "ink"):
            out[key].append(j[key])
        out["t_fwd"].append(prob[0, j["pr"] - R0, j["pc"] - C0]); out["t_rev"].append(prob[1, j["pr"] - R0, j["pc"] - C0])
        print(j["seg"], "chunk", j["n"], "points", k, "windows", len(wins), f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(args.out or ("wt_points.npz" if not only else f"wt_points_{'_'.join(only)}.npz"),
                        **{k: np.concatenate(v) for k, v in out.items()})
    print("saved")


if __name__ == "__main__":
    main()
