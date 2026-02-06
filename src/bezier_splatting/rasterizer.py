"""Tile-based differentiable 2D Gaussian splatting rasterizer.

Pure PyTorch implementation — no CUDA kernels. Prioritizes clarity and
correctness over raw performance. Targets 256×256 resolution.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from .sampling import GaussianParams


def _build_covariance(
    scales: Float[Tensor, "G 2"],
    rotations: Float[Tensor, " G"],
) -> Float[Tensor, "G 2 2"]:
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


def _invert_2x2(cov: Float[Tensor, "G 2 2"]) -> tuple[Float[Tensor, "G 2 2"], Float[Tensor, " G"]]:
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
    bg_color: Float[Tensor, " 3"] | None = None,
    tile_size: int = 16,
    chunk_size: int = 16,
) -> Float[Tensor, "3 H W"]:
    """Tile-based differentiable 2D Gaussian splatting.

    Algorithm:
        1. Compute covariance matrices and bounding boxes
        2. Assign Gaussians to tiles (vectorized GPU-side)
        3. Chunked batched alpha compositing across tiles
        4. Stitch tiles into final image

    Args:
        gaussians: GaussianParams with G total Gaussians
        H, W: Image height and width in pixels
        bg_color: Background color (3,). Defaults to white.
        tile_size: Tile size in pixels. Default 16.
        chunk_size: Number of tiles to process in parallel per chunk. Default 16.

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

    # ── 2. Vectorized tile assignment ─────────────────────────────────
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles = n_tiles_x * n_tiles_y

    tile_min_x = torch.clamp((bb_min_x / tile_size).floor().int(), 0, n_tiles_x - 1)
    tile_min_y = torch.clamp((bb_min_y / tile_size).floor().int(), 0, n_tiles_y - 1)
    tile_max_x = torch.clamp((bb_max_x / tile_size).ceil().int(), 0, n_tiles_x - 1)
    tile_max_y = torch.clamp((bb_max_y / tile_size).ceil().int(), 0, n_tiles_y - 1)

    tile_ys = torch.arange(n_tiles_y, device=device)
    tile_xs = torch.arange(n_tiles_x, device=device)
    in_y = (tile_min_y[:, None] <= tile_ys) & (tile_ys <= tile_max_y[:, None])  # (G, TY)
    in_x = (tile_min_x[:, None] <= tile_xs) & (tile_xs <= tile_max_x[:, None])  # (G, TX)
    membership = (in_y[:, :, None] & in_x[:, None, :])  # (G, TY, TX)
    membership = membership.reshape(G, n_tiles)  # (G, T)

    gaussians_per_tile = membership.sum(dim=0)  # (T,)

    # ── 3. Pre-build pixel grid ───────────────────────────────────────
    opacities = torch.sigmoid(gaussians.opacities)  # (G,)
    colors = gaussians.colors  # (G, 3)

    px_x = torch.arange(W, device=device, dtype=dtype) + 0.5
    px_y = torch.arange(H, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(px_y, px_x, indexing="ij")
    pixel_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

    # ── 4. Chunked batched rendering ──────────────────────────────────
    image = torch.zeros(3, H, W, device=device, dtype=dtype)

    tile_order = torch.argsort(gaussians_per_tile)  # ascending by Gaussian count
    tile_start = 0

    while tile_start < n_tiles:
        tile_end = min(tile_start + chunk_size, n_tiles)
        chunk_tile_ids = tile_order[tile_start:tile_end]  # flat tile indices in this chunk
        n_chunk = chunk_tile_ids.shape[0]

        chunk_counts = gaussians_per_tile[chunk_tile_ids]  # (n_chunk,)
        max_T = int(chunk_counts.max().item())

        if max_T == 0:
            for ci in range(n_chunk):
                flat_tid = chunk_tile_ids[ci].item()
                ty = flat_tid // n_tiles_x
                tx = flat_tid % n_tiles_x
                px_sy = ty * tile_size
                px_sx = tx * tile_size
                px_ey = min(px_sy + tile_size, H)
                px_ex = min(px_sx + tile_size, W)
                image[:, px_sy:px_ey, px_sx:px_ex] = bg_color[:, None, None]
            tile_start = tile_end
            continue

        # Build padded index tensors for the chunk: (n_chunk, max_T)
        # Membership columns for the chunk tiles: (G, n_chunk)
        chunk_membership = membership[:, chunk_tile_ids]  # (G, n_chunk)

        # For each tile in the chunk, collect sorted Gaussian indices
        # Use argsort-based approach: for each tile column, nonzero gives indices
        padded_idx = torch.zeros(n_chunk, max_T, device=device, dtype=torch.long)
        valid_mask = torch.zeros(n_chunk, max_T, device=device, dtype=torch.bool)

        for ci in range(n_chunk):
            g_ids = chunk_membership[:, ci].nonzero(as_tuple=False).squeeze(-1)
            n_g = g_ids.shape[0]
            if n_g > 0:
                padded_idx[ci, :n_g] = g_ids
                valid_mask[ci, :n_g] = True

        # Gather Gaussian data: (n_chunk, max_T, ...)
        flat_idx = padded_idx.reshape(-1)  # (n_chunk * max_T,)
        g_means = means[flat_idx].reshape(n_chunk, max_T, 2)
        g_inv_cov = inv_cov[flat_idx].reshape(n_chunk, max_T, 2, 2)
        g_opacities = opacities[flat_idx].reshape(n_chunk, max_T)
        g_colors = colors[flat_idx].reshape(n_chunk, max_T, 3)

        # Zero out padding opacities so they don't contribute
        g_opacities = g_opacities * valid_mask.float()

        # Collect pixel grids for each tile in the chunk: (n_chunk, tile_size, tile_size, 2)
        # Edge tiles may be smaller; pad to tile_size × tile_size
        chunk_pixels = torch.zeros(n_chunk, tile_size, tile_size, 2, device=device, dtype=dtype)
        tile_heights = []
        tile_widths = []

        for ci in range(n_chunk):
            flat_tid = chunk_tile_ids[ci].item()
            ty = flat_tid // n_tiles_x
            tx = flat_tid % n_tiles_x
            px_sy = ty * tile_size
            px_sx = tx * tile_size
            px_ey = min(px_sy + tile_size, H)
            px_ex = min(px_sx + tile_size, W)
            th = px_ey - px_sy
            tw = px_ex - px_sx
            tile_heights.append(th)
            tile_widths.append(tw)
            chunk_pixels[ci, :th, :tw, :] = pixel_grid[px_sy:px_ey, px_sx:px_ex, :]

        # Displacement: (n_chunk, max_T, tile_size, tile_size, 2)
        d = chunk_pixels[:, None, :, :, :] - g_means[:, :, None, None, :]

        # Mahalanobis: einsum over the 2D displacement with inverse covariance
        # d: (C, T, th, tw, 2), g_inv_cov: (C, T, 2, 2) → d_transformed: (C, T, th, tw, 2)
        d_transformed = torch.einsum("ctpqi,ctij->ctpqj", d, g_inv_cov)
        mahal = 0.5 * (d * d_transformed).sum(dim=-1)  # (n_chunk, max_T, tile_size, tile_size)

        # Alpha: (n_chunk, max_T, tile_size, tile_size)
        alpha = g_opacities[:, :, None, None] * torch.exp(-mahal)
        alpha = torch.clamp(alpha, 0.0, 0.99)

        # Front-to-back compositing along the Gaussian dimension (dim=1)
        one_minus_alpha = 1.0 - alpha  # (n_chunk, max_T, tile_size, tile_size)

        transmittance = torch.ones_like(alpha)
        if max_T > 1:
            transmittance[:, 1:] = torch.cumprod(one_minus_alpha[:, :-1], dim=1)

        weights = alpha * transmittance  # (n_chunk, max_T, tile_size, tile_size)

        # Weighted color: g_colors (n_chunk, max_T, 3), weights (n_chunk, max_T, tile_size, tile_size)
        rendered = torch.einsum("ntk,ntpq->nkpq", g_colors, weights)

        # Remaining transmittance via reuse of cumprod result
        remaining_T = transmittance[:, -1:] * one_minus_alpha[:, -1:]  # (n_chunk, 1, tile_size, tile_size)
        remaining_T = remaining_T.squeeze(1)  # (n_chunk, tile_size, tile_size)

        rendered = rendered + bg_color[None, :, None, None] * remaining_T[:, None, :, :]

        # Write results back to the output image
        for ci in range(n_chunk):
            flat_tid = chunk_tile_ids[ci].item()
            ty = flat_tid // n_tiles_x
            tx = flat_tid % n_tiles_x
            px_sy = ty * tile_size
            px_sx = tx * tile_size
            th = tile_heights[ci]
            tw = tile_widths[ci]
            image[:, px_sy:px_sy + th, px_sx:px_sx + tw] = rendered[ci, :, :th, :tw]

        tile_start = tile_end

    return image
