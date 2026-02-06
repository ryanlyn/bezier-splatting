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
    scale = torch.tensor([W, H], device=cp.device, dtype=cp.dtype)
    return (cp + 1) / 2 * scale


def pixel_to_model(cp: Float[Tensor, "*batch 2"], H: int, W: int) -> Float[Tensor, "*batch 2"]:
    """Convert pixel coords to [-1, 1] model coords."""
    scale = torch.tensor([W, H], device=cp.device, dtype=cp.dtype)
    return cp / scale * 2 - 1


def legacy_to_model(cp: Float[Tensor, "*batch 2"]) -> Float[Tensor, "*batch 2"]:
    """Convert legacy [0, 1] coords to [-1, 1] model coords."""
    return cp * 2 - 1
