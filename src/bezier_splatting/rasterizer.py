"""Tile-based differentiable 2D Gaussian splatting rasterizer.

Pure PyTorch implementation — no CUDA kernels. Prioritizes clarity and
correctness over raw performance. Targets 256×256 resolution.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .sampling import GaussianParams


def _build_covariance(scales: Tensor, rotations: Tensor) -> Tensor:
    """Build 2×2 covariance matrices from scales and rotations.

    Σ = R @ diag(σ²) @ R^T

    Args:
        scales: (G, 2) — [σ_x, σ_y]
        rotations: (G,) — angle θ in radians

    Returns:
        Covariance matrices (G, 2, 2).
    """
    cos = torch.cos(rotations)  # (G,)
    sin = torch.sin(rotations)

    # Rotation matrix columns
    # R = [[cos, -sin], [sin, cos]]
    # Σ = R @ diag(σ²) @ R^T
    sx2 = scales[:, 0] ** 2  # σ_x²
    sy2 = scales[:, 1] ** 2  # σ_y²

    # Expanded: Σ = [[cos²σx² + sin²σy², cossin(σx²-σy²)],
    #                [cossin(σx²-σy²), sin²σx² + cos²σy²]]
    a = cos ** 2 * sx2 + sin ** 2 * sy2
    b = cos * sin * (sx2 - sy2)
    d = sin ** 2 * sx2 + cos ** 2 * sy2

    cov = torch.stack([
        torch.stack([a, b], dim=-1),
        torch.stack([b, d], dim=-1),
    ], dim=-2)  # (G, 2, 2)

    return cov


def _invert_2x2(cov: Tensor) -> tuple[Tensor, Tensor]:
    """Analytic inverse of 2×2 symmetric positive-definite matrices.

    [[a, b], [b, d]]⁻¹ = (1/det) * [[d, -b], [-b, a]]

    Args:
        cov: (G, 2, 2) covariance matrices

    Returns:
        (inv_cov, det) — inverse covariance (G, 2, 2) and determinant (G,)
    """
    a = cov[:, 0, 0]
    b = cov[:, 0, 1]
    d = cov[:, 1, 1]

    det = a * d - b * b
    det = torch.clamp(det, min=1e-8)  # numerical safety

    inv_det = 1.0 / det
    inv_cov = torch.stack([
        torch.stack([d * inv_det, -b * inv_det], dim=-1),
        torch.stack([-b * inv_det, a * inv_det], dim=-1),
    ], dim=-2)

    return inv_cov, det


def rasterize(
    gaussians: GaussianParams,
    H: int,
    W: int,
    bg_color: Tensor | None = None,
    tile_size: int = 16,
) -> Tensor:
    """Tile-based differentiable 2D Gaussian splatting.

    Algorithm:
        1. Compute covariance matrices and bounding boxes
        2. Assign Gaussians to tiles
        3. Per-tile front-to-back alpha compositing
        4. Stitch tiles into final image

    Args:
        gaussians: GaussianParams with G total Gaussians
        H, W: Image height and width in pixels
        bg_color: Background color (3,). Defaults to white.
        tile_size: Tile size in pixels. Default 16.

    Returns:
        Rendered image (3, H, W) in [0, 1].
    """
    device = gaussians.means.device
    dtype = gaussians.means.dtype
    G = gaussians.means.shape[0]

    if bg_color is None:
        bg_color = torch.ones(3, device=device, dtype=dtype)

    if G == 0:
        return bg_color[:, None, None].expand(3, H, W).clone()

    # ── 1. Covariance matrices and bounding boxes ─────────────────────
    cov = _build_covariance(gaussians.scales, gaussians.rotations)  # (G, 2, 2)
    inv_cov, det = _invert_2x2(cov)  # (G, 2, 2), (G,)

    # 3σ bounding radius in x and y
    # For an axis-aligned bound, use sqrt of diagonal elements of Σ
    radius_x = 3.0 * torch.sqrt(cov[:, 0, 0])  # (G,)
    radius_y = 3.0 * torch.sqrt(cov[:, 1, 1])

    means = gaussians.means  # (G, 2)
    # Bounding box in pixel coords: [x_min, y_min, x_max, y_max]
    bb_min_x = means[:, 0] - radius_x
    bb_min_y = means[:, 1] - radius_y
    bb_max_x = means[:, 0] + radius_x
    bb_max_y = means[:, 1] + radius_y

    # ── 2. Tile assignment ────────────────────────────────────────────
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size

    # For each Gaussian, compute which tile range it covers
    tile_min_x = torch.clamp((bb_min_x / tile_size).floor().int(), 0, n_tiles_x - 1)
    tile_min_y = torch.clamp((bb_min_y / tile_size).floor().int(), 0, n_tiles_y - 1)
    tile_max_x = torch.clamp((bb_max_x / tile_size).ceil().int(), 0, n_tiles_x - 1)
    tile_max_y = torch.clamp((bb_max_y / tile_size).ceil().int(), 0, n_tiles_y - 1)

    # Build tile→Gaussian mapping
    # For moderate G (< ~10K), building a list per tile is fine
    tile_gaussian_lists: list[list[int]] = [[] for _ in range(n_tiles_x * n_tiles_y)]

    tile_min_x_cpu = tile_min_x.cpu().tolist()
    tile_min_y_cpu = tile_min_y.cpu().tolist()
    tile_max_x_cpu = tile_max_x.cpu().tolist()
    tile_max_y_cpu = tile_max_y.cpu().tolist()

    for g_idx in range(G):
        for ty in range(tile_min_y_cpu[g_idx], tile_max_y_cpu[g_idx] + 1):
            for tx in range(tile_min_x_cpu[g_idx], tile_max_x_cpu[g_idx] + 1):
                tile_gaussian_lists[ty * n_tiles_x + tx].append(g_idx)

    # ── 3. Per-tile rendering ─────────────────────────────────────────
    # Precompute sigmoid opacities
    opacities = torch.sigmoid(gaussians.opacities)  # (G,)
    colors = gaussians.colors  # (G, 3)

    # Output image
    image = torch.zeros(3, H, W, device=device, dtype=dtype)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_idx = ty * n_tiles_x + tx
            g_indices = tile_gaussian_lists[tile_idx]

            # Tile pixel range
            px_start_x = tx * tile_size
            px_start_y = ty * tile_size
            px_end_x = min(px_start_x + tile_size, W)
            px_end_y = min(px_start_y + tile_size, H)
            tw = px_end_x - px_start_x
            th = px_end_y - px_start_y

            if len(g_indices) == 0:
                # No Gaussians → background color
                image[:, px_start_y:px_end_y, px_start_x:px_end_x] = bg_color[:, None, None]
                continue

            # Gather Gaussian data for this tile
            idx = torch.tensor(g_indices, device=device, dtype=torch.long)
            tile_means = means[idx]        # (T, 2)
            tile_inv_cov = inv_cov[idx]    # (T, 2, 2)
            tile_opacities = opacities[idx]  # (T,)
            tile_colors = colors[idx]      # (T, 3)
            T_count = len(g_indices)

            # Build pixel grid for this tile
            px_x = torch.arange(px_start_x, px_end_x, device=device, dtype=dtype) + 0.5
            px_y = torch.arange(px_start_y, px_end_y, device=device, dtype=dtype) + 0.5
            grid_y, grid_x = torch.meshgrid(px_y, px_x, indexing="ij")
            pixels = torch.stack([grid_x, grid_y], dim=-1)  # (th, tw, 2)

            # Displacement from each pixel to each Gaussian mean
            # pixels: (th, tw, 2), tile_means: (T, 2) → d: (T, th, tw, 2)
            d = pixels[None, :, :, :] - tile_means[:, None, None, :]

            # Mahalanobis distance: 0.5 * d^T Σ⁻¹ d
            # d: (T, th, tw, 2), inv_cov: (T, 2, 2) → (T, th, tw)
            d_transformed = torch.einsum("tpqi,tij->tpqj", d, tile_inv_cov)
            mahal = 0.5 * (d * d_transformed).sum(dim=-1)  # (T, th, tw)

            # Alpha values: α = opacity * exp(-mahal), clamped to [0, 0.99]
            alpha = tile_opacities[:, None, None] * torch.exp(-mahal)
            alpha = torch.clamp(alpha, 0.0, 0.99)

            # Front-to-back alpha compositing
            # C = Σ c_i * α_i * Π_{j<i}(1 - α_j)
            # Transmittance: T_i = Π_{j<i}(1 - α_j)
            # Using cumulative product of (1 - α)
            one_minus_alpha = 1.0 - alpha  # (T, th, tw)

            # Cumulative product shifted by one (T_0 = 1, T_1 = 1-α_0, ...)
            # Use cumprod along the Gaussian dimension (dim=0)
            transmittance = torch.ones_like(alpha)
            if T_count > 1:
                transmittance[1:] = torch.cumprod(one_minus_alpha[:-1], dim=0)

            # Weight for each Gaussian: w_i = α_i * T_i
            weights = alpha * transmittance  # (T, th, tw)

            # Weighted color sum
            # tile_colors: (T, 3), weights: (T, th, tw)
            rendered = torch.einsum("tc,tpq->cpq", tile_colors, weights)  # (3, th, tw)

            # Add background: remaining transmittance * bg_color
            remaining_T = torch.prod(one_minus_alpha, dim=0)  # (th, tw)
            rendered = rendered + bg_color[:, None, None] * remaining_T[None, :, :]

            image[:, px_start_y:px_end_y, px_start_x:px_end_x] = rendered

    return image
