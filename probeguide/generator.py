"""ProbeGuidedLabelGenerator: villa's DinoGuidedLabelGenerator with a linear-probe similarity map.

villa (`vesuvius/ink_detection/training/dynamic_labels.py`) builds 3D ink pseudo-labels as

    label = (frozen ink-UNet probability * S) > threshold

where S is the cosine similarity of every Dinovol patch token to ONE reference ink embedding (`avg_ref_embedding.npy`,
the mean of 128 unique expert-clicked tokens), blended over 128^3 windows (no-op at the team's dino_stride 128) and
min-max normalised per 256^3 chunk.

ProbeGuide keeps the backbone, windows, stride, blending and upsampling (tests/test_generator.py proves the geometry
matches the parent exactly) and only replaces the per-token score by a linear probe on the L2-normalised tokens:

    S = sigmoid(normalize(token) . coef + intercept)        (no per-chunk min-max)

Probe weights: `results/probe_surface.npz` (inside the recipe, where the UNet masks air) and `results/probe_with_air.npz`
(standalone). They are class-balanced logistic regressions fit from the team's 2D ink labels mapped to 3D (not
calibrated to the true ink rate). See README for the held-out evaluation and for when the gate helps at all.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.ink_detection.training.dynamic_labels import (  # villa's own implementation
    DinoGridSpec,
    DinoGuidedLabelGenerator,
    _config_alias,
    _config_value,
    load_frozen_dino_backbone,
    load_frozen_ink_model,
)


def load_probe(path: str, embedding_dim: int = 864) -> tuple[np.ndarray, float]:
    d = np.load(path)
    coef = np.asarray(d["coef"], dtype=np.float32)
    if coef.shape != (embedding_dim,):
        raise ValueError(f"probe coef must have shape ({embedding_dim},), got {coef.shape}")
    return coef, float(d["intercept"])


class ProbeGuidedLabelGenerator(DinoGuidedLabelGenerator):
    """Same interface and label rule as DinoGuidedLabelGenerator; S comes from a linear probe (no min-max)."""

    def __init__(self, *, probe_coef: np.ndarray, probe_intercept: float, **kwargs: Any) -> None:
        dim = (kwargs.get("grid") or DinoGridSpec()).embedding_dim
        # the parent requires a reference embedding; it is unused here
        super().__init__(reference_embedding=torch.zeros(dim) + 1.0, **kwargs)
        self.probe_coef = torch.as_tensor(probe_coef, dtype=torch.float32, device=self.device)
        self.probe_intercept = float(probe_intercept)

    @classmethod
    def from_mapping(cls, config: Any, *, device="cuda", dtype=torch.bfloat16) -> "ProbeGuidedLabelGenerator":
        grid = DinoGridSpec()
        coef, intercept = load_probe(_config_alias(config, "probe_weights", "probe"), grid.embedding_dim)
        return cls(
            probe_coef=coef,
            probe_intercept=intercept,
            unet=load_frozen_ink_model(_config_alias(config, "unet_ckpt", "unet_checkpoint"), device=device, dtype=dtype),
            dino=load_frozen_dino_backbone(_config_alias(config, "dino_ckpt", "dino_checkpoint"), device=device, dtype=dtype),
            device=device,
            dtype=dtype,
            dino_stride=int(_config_value(config, "dino_stride", 64)),
            dino_minibatch=int(_config_value(config, "dino_minibatch", 8)),
            dino_blend_sigma=float(_config_value(config, "dino_blend_sigma", 4.0)),
            threshold=float(_config_value(config, "threshold", 0.5)),
            grid=grid,
        )

    def token_scores(self, tokens: torch.Tensor) -> torch.Tensor:
        """(n, t, 864) patch tokens -> (n, t) probe probabilities."""
        return torch.sigmoid(F.normalize(tokens.float(), dim=-1) @ self.probe_coef + self.probe_intercept)

    @torch.inference_mode()
    def _dino_similarity(self, image_b1zyx: torch.Tensor) -> torch.Tensor:
        # Identical window/blend/upsample logic to the parent; only the per-token score and the absence of
        # per-chunk min-max differ.
        batch_size = int(image_b1zyx.shape[0])
        grid = self.grid
        coordinates = [(z, y, x) for z in self.starts for y in self.starts for x in self.starts]
        t = grid.tokens_per_window_axis
        acc = torch.zeros((batch_size,) + (grid.tokens_per_chunk_axis,) * 3, device=self.device, dtype=torch.float32)
        wacc = torch.zeros_like(acc)
        for start in range(0, len(coordinates), self.dino_minibatch):
            batch_coords = coordinates[start:start + self.dino_minibatch]
            windows = torch.cat([
                image_b1zyx[:, :, z0:z0 + grid.window_size, y0:y0 + grid.window_size, x0:x0 + grid.window_size]
                for z0, y0, x0 in batch_coords
            ]).to(device=self.device, dtype=self.dtype)
            tokens = self.dino.forward_features(windows)["x_norm_patchtokens"]
            scores = self.token_scores(tokens).reshape(len(batch_coords), batch_size, t, t, t)
            for j, (z0, y0, x0) in enumerate(batch_coords):
                oz, oy, ox = z0 // grid.patch_size, y0 // grid.patch_size, x0 // grid.patch_size
                sl = (slice(None), slice(oz, oz + t), slice(oy, oy + t), slice(ox, ox + t))
                acc[sl] += scores[j] * self.weight
                wacc[sl] += self.weight
        grid_scores = acc / wacc.clamp_min(1e-6)
        return F.interpolate(grid_scores.unsqueeze(1), size=(grid.chunk_size,) * 3, mode="trilinear",
                             align_corners=False)
