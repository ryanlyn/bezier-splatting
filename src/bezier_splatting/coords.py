"""Coordinate space conversion helpers.

Model space: [-1, 1] normalized coordinates (resolution-independent).
Pixel space: [0, W] x [0, H] (used by samplers and rasterizer).
Legacy space: [0, 1] (used by older checkpoints).
"""

import torch
from jaxtyping import Float
from torch import Tensor


def model_to_pixel(cp: Float[Tensor, "*batch 2"], H: int, W: int) -> Float[Tensor, "*batch 2"]:
    """Convert [-1, 1] model coords to pixel coords."""
    out = torch.empty_like(cp)
    out[..., 0] = (cp[..., 0] + 1.0) * (0.5 * W)
    out[..., 1] = (cp[..., 1] + 1.0) * (0.5 * H)
    return out


def pixel_to_model(cp: Float[Tensor, "*batch 2"], H: int, W: int) -> Float[Tensor, "*batch 2"]:
    """Convert pixel coords to [-1, 1] model coords."""
    out = torch.empty_like(cp)
    out[..., 0] = cp[..., 0] * (2.0 / W) - 1.0
    out[..., 1] = cp[..., 1] * (2.0 / H) - 1.0
    return out


def legacy_to_model(cp: Float[Tensor, "*batch 2"]) -> Float[Tensor, "*batch 2"]:
    """Convert legacy [0, 1] coords to [-1, 1] model coords."""
    return cp * 2 - 1
