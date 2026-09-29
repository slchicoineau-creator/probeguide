"""Minimal, dependency-light reader for remote Zarr v2 arrays (HTTP), with a local chunk cache.

Works for the Vesuvius open-data S3 bucket and the Hugging Face `scrollprize/datasets` bucket.
Only what the benchmark needs: read an axis-aligned box [z0:z1, y0:y1, x0:x1] of one pyramid level.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve"
CACHE = Path(os.environ.get("SB_CACHE", Path(__file__).resolve().parents[1] / "cache"))

_lock = threading.Lock()


def _get(url: str, retries: int = 4) -> bytes | None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None  # missing chunk -> fill value
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
    return None


class RemoteArray:
    """One level of a remote Zarr v2 array."""

    def __init__(self, base_url: str, level: int | str = 0):
        self.url = f"{base_url.rstrip('/')}/{level}"
        meta = json.loads(_get(self.url + "/.zarray"))
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill = meta["fill_value"] if meta["fill_value"] is not None else 0
        self.sep = meta.get("dimension_separator", ".")
        comp = meta.get("compressor")
        self.codec = numcodecs.get_codec(comp) if comp else None
        key = self.url.replace("https://", "").replace("/", "_").replace(":", "")
        self.cache_dir = CACHE / key
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _chunk(self, idx: tuple[int, int, int]) -> np.ndarray:
        name = self.sep.join(map(str, idx))
        f = self.cache_dir / name.replace("/", "_")
        if f.exists():
            raw = f.read_bytes()
        else:
            raw = _get(f"{self.url}/{name}")
            raw = b"" if raw is None else raw
            f.write_bytes(raw)
        if not raw:
            return np.full(self.chunks, self.fill, dtype=self.dtype)
        buf = self.codec.decode(raw) if self.codec else raw
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.chunks)

    def read(self, z0, z1, y0, y1, x0, x1, workers: int = 16) -> np.ndarray:
        lo = (z0, y0, x0)
        hi = (z1, y1, x1)
        out = np.full([h - l for l, h in zip(lo, hi)], self.fill, dtype=self.dtype)
        rng = [range(max(l, 0) // c, (min(h, s) - 1) // c + 1)
               for l, h, c, s in zip(lo, hi, self.chunks, self.shape)]
        jobs = [(a, b, c) for a in rng[0] for b in rng[1] for c in rng[2]]

        def load(ix):
            ch = self._chunk(ix)
            sl_src, sl_dst = [], []
            for d in range(3):
                c0 = ix[d] * self.chunks[d]
                s = max(lo[d], c0)
                e = min(hi[d], c0 + self.chunks[d], self.shape[d])
                sl_src.append(slice(s - c0, e - c0))
                sl_dst.append(slice(s - lo[d], e - lo[d]))
            out[tuple(sl_dst)] = ch[tuple(sl_src)]

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(load, jobs))
        return out


# CT store used by the evaluation (PHerc. Paris 4 = Scroll 1, 2.4 um scan 20260411134726)
CT_2UM = f"{S3}/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
