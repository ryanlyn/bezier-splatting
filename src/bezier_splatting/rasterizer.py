"""Tile-based differentiable 2D Gaussian splatting rasterizer.

Provides two backends:
- ``pytorch``: pure-PyTorch implementation that works on all devices (CPU, CUDA, MPS).
- ``gsplat``: CUDA-accelerated backend using the gsplat library (requires CUDA device).

Legacy backend strings ``"reference"`` and ``"mps"`` are accepted as aliases for ``"pytorch"``.

Both backends are differentiable and return identical shapes.
"""

import warnings

import torch
from jaxtyping import Float
from torch import Tensor

from .sampling import GaussianParams

RasterBackend = str  # Literal["auto", "pytorch", "gsplat", "reference", "mps"]


def _check_gsplat() -> bool:
    """Check whether the gsplat library is importable."""
    try:
        import gsplat  # noqa: F401
        return True
    except ImportError:
        return False


def _build_covariance(
    scales: Float[Tensor, "G 2"],
    rotations: Float[Tensor, " G"],
) -> Float[Tensor, "G 2 2"]:
    """Build 2x2 covariance matrices from scales and rotations.

    Sigma = R @ diag(sigma^2) @ R^T

    Args:
        scales: (G, 2) -- [sigma_x, sigma_y]
        rotations: (G,) -- angle theta in radians

    Returns:
        Covariance matrices (G, 2, 2).
    """
    cos = torch.cos(rotations)  # (G,)
    sin = torch.sin(rotations)

    # Rotation matrix columns
    # R = [[cos, -sin], [sin, cos]]
    # Sigma = R @ diag(sigma^2) @ R^T
    sx2 = scales[:, 0] ** 2  # sigma_x^2
    sy2 = scales[:, 1] ** 2  # sigma_y^2

    # Expanded: Sigma = [[cos^2*sx2 + sin^2*sy2, cos*sin*(sx2-sy2)],
    #                     [cos*sin*(sx2-sy2), sin^2*sx2 + cos^2*sy2]]
    a = cos ** 2 * sx2 + sin ** 2 * sy2
    b = cos * sin * (sx2 - sy2)
    d = sin ** 2 * sx2 + cos ** 2 * sy2

    cov = torch.stack([
        torch.stack([a, b], dim=-1),
        torch.stack([b, d], dim=-1),
    ], dim=-2)  # (G, 2, 2)

    return cov


def _invert_2x2(cov: Float[Tensor, "G 2 2"]) -> tuple[Float[Tensor, "G 2 2"], Float[Tensor, " G"]]:
    """Analytic inverse of 2x2 symmetric positive-definite matrices.

    [[a, b], [b, d]]^-1 = (1/det) * [[d, -b], [-b, a]]

    Args:
        cov: (G, 2, 2) covariance matrices

    Returns:
        (inv_cov, det) -- inverse covariance (G, 2, 2) and determinant (G,)
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


def _resolve_backend(backend: RasterBackend, device: torch.device) -> str:
    """Resolve backend choice based on tensor device and library availability.

    Returns ``"pytorch"`` or ``"gsplat"``.
    """
    if backend == "auto":
        if device.type == "cuda":
            if _check_gsplat():
                return "gsplat"
            warnings.warn(
                "Raster backend `auto` resolved to `pytorch` on CUDA because `gsplat` is unavailable. "
                "This can be significantly slower and may use more memory. "
                "Install `gsplat` (e.g. `uv sync --extra cuda` or `pip install gsplat`) "
                "or set backend explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )
        return "pytorch"

    if backend in {"pytorch", "mps", "reference"}:
        return "pytorch"

    if backend in {"gsplat", "cuda_gsplat"}:
        if device.type != "cuda":
            raise ValueError(
                f"gsplat backend requires a CUDA device, got {device.type!r}",
            )
        if not _check_gsplat():
            raise ImportError(
                "gsplat backend requested but the gsplat library is not installed. "
                "Install it with: pip install gsplat",
            )
        return "gsplat"

    raise ValueError(
        f"Unknown raster backend: {backend!r} "
        f"(expected 'auto', 'pytorch', 'gsplat', 'reference', or 'mps')",
    )


def _default_pytorch_chunk_size(
    H: int,
    W: int,
    tile_size: int,
    n_gaussians: int,
) -> int:
    """Choose a robust chunk size for common workload regimes."""
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles = n_tiles_x * n_tiles_y

    if n_tiles <= 64:
        return n_tiles

    density = n_gaussians / float(n_tiles)

    # Moderate density scenes benefit from fewer chunk transitions, but very
    # dense scenes pay more for max_T padding when chunks get too large.
    if density <= 20.0:
        return min(96, n_tiles)

    return min(64, n_tiles)


def _rasterize_pytorch(
    gaussians: GaussianParams,
    H: int,
    W: int,
    bg_color: Float[Tensor, " 3"] | None = None,
    tile_size: int = 16,
    chunk_size: int = 64,
) -> Float[Tensor, "3 H W"]:
    """Pure-PyTorch tile rasterizer that works on all devices (CPU, CUDA, MPS).

    Design goals:
    - Keep computation on device (avoid frequent host syncs).
    - Remove Python per-tile loops in Gaussian gather/writeback.
    - Preserve front-to-back compositing order and differentiability.
    """
    device = gaussians.means.device
    dtype = gaussians.means.dtype
    G = gaussians.means.shape[0]

    if bg_color is None:
        bg_color = torch.ones(3, device=device, dtype=dtype)

    if G == 0:
        return bg_color[:, None, None].expand(3, H, W).clone()

    cov = _build_covariance(gaussians.scales, gaussians.rotations)
    inv_cov, _ = _invert_2x2(cov)

    radius_x = 3.0 * torch.sqrt(cov[:, 0, 0])
    radius_y = 3.0 * torch.sqrt(cov[:, 1, 1])

    means = gaussians.means
    bb_min_x = means[:, 0] - radius_x
    bb_min_y = means[:, 1] - radius_y
    bb_max_x = means[:, 0] + radius_x
    bb_max_y = means[:, 1] + radius_y

    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles = n_tiles_x * n_tiles_y

    tile_min_x = torch.clamp((bb_min_x / tile_size).floor().long(), 0, n_tiles_x - 1)
    tile_min_y = torch.clamp((bb_min_y / tile_size).floor().long(), 0, n_tiles_y - 1)
    tile_max_x = torch.clamp((bb_max_x / tile_size).ceil().long(), 0, n_tiles_x - 1)
    tile_max_y = torch.clamp((bb_max_y / tile_size).ceil().long(), 0, n_tiles_y - 1)

    tile_ys = torch.arange(n_tiles_y, device=device, dtype=torch.long)
    tile_xs = torch.arange(n_tiles_x, device=device, dtype=torch.long)
    in_y = (tile_min_y[:, None] <= tile_ys) & (tile_ys <= tile_max_y[:, None])
    in_x = (tile_min_x[:, None] <= tile_xs) & (tile_xs <= tile_max_x[:, None])
    membership = (in_y[:, :, None] & in_x[:, None, :]).reshape(G, n_tiles)  # (G, T)

    gaussians_per_tile = membership.sum(dim=0)  # (T,)

    opacities = torch.sigmoid(gaussians.opacities)
    colors = gaussians.colors

    # Use a padded tile-aligned backing image so tile writes stay contiguous.
    # This avoids expensive per-pixel scatter on MPS.
    H_pad = n_tiles_y * tile_size
    W_pad = n_tiles_x * tile_size
    image_pad = bg_color[:, None, None].expand(3, H_pad, W_pad).clone()
    image_tiles = image_pad.view(3, n_tiles_y, tile_size, n_tiles_x, tile_size).permute(1, 3, 0, 2, 4)

    # Process denser tiles first to reduce repeated allocations from large max_T.
    tile_order = torch.argsort(gaussians_per_tile, descending=True)

    # Local tile pixel grid (integer offsets).
    ys_i = torch.arange(tile_size, device=device, dtype=torch.long)
    xs_i = torch.arange(tile_size, device=device, dtype=torch.long)
    local_y_i, local_x_i = torch.meshgrid(ys_i, xs_i, indexing="ij")
    local_y = local_y_i.to(dtype)
    local_x = local_x_i.to(dtype)

    for tile_start in range(0, n_tiles, chunk_size):
        chunk_tile_ids = tile_order[tile_start:tile_start + chunk_size]
        n_chunk = chunk_tile_ids.shape[0]
        if n_chunk == 0:
            continue

        chunk_counts = gaussians_per_tile[chunk_tile_ids]
        max_T = int(chunk_counts.max().item())
        if max_T == 0:
            # These tiles remain background.
            continue

        chunk_membership = membership[:, chunk_tile_ids]  # (G, C)

        # Compute insertion slot per (gaussian, tile) pair with cumulative counts.
        # slot[g, c] is valid only where chunk_membership[g, c] is True.
        slot = chunk_membership.to(torch.int32).cumsum(dim=0) - 1

        g_idx, c_idx = chunk_membership.nonzero(as_tuple=True)
        slot_idx = slot[g_idx, c_idx]

        padded_idx = torch.zeros(n_chunk, max_T, device=device, dtype=torch.long)
        valid_mask = torch.zeros(n_chunk, max_T, device=device, dtype=torch.bool)
        padded_idx[c_idx, slot_idx] = g_idx
        valid_mask[c_idx, slot_idx] = True

        flat_idx = padded_idx.reshape(-1)
        g_means = means[flat_idx].reshape(n_chunk, max_T, 2)
        g_inv_cov = inv_cov[flat_idx].reshape(n_chunk, max_T, 2, 2)
        g_opacities = opacities[flat_idx].reshape(n_chunk, max_T)
        g_colors = colors[flat_idx].reshape(n_chunk, max_T, 3)

        g_opacities = g_opacities * valid_mask.to(dtype)

        ty = torch.div(chunk_tile_ids, n_tiles_x, rounding_mode="floor")
        tx = chunk_tile_ids % n_tiles_x
        tile_x0 = tx * tile_size
        tile_y0 = ty * tile_size

        # Chunk pixel coordinates: (C, TS, TS)
        px_x = tile_x0[:, None, None].to(dtype) + local_x[None, :, :] + 0.5
        px_y = tile_y0[:, None, None].to(dtype) + local_y[None, :, :] + 0.5

        d_x = px_x[:, None, :, :] - g_means[:, :, None, None, 0]
        d_y = px_y[:, None, :, :] - g_means[:, :, None, None, 1]

        # For inverse covariance [[a, b], [b, d]]:
        # 0.5 * [dx,dy] Sigma^-1 [dx,dy]^T = 0.5 * (a*dx^2 + 2b*dx*dy + d*dy^2)
        a = g_inv_cov[:, :, 0, 0][:, :, None, None]
        b = g_inv_cov[:, :, 0, 1][:, :, None, None]
        d = g_inv_cov[:, :, 1, 1][:, :, None, None]
        mahal = 0.5 * (a * d_x * d_x + 2.0 * b * d_x * d_y + d * d_y * d_y)

        alpha = g_opacities[:, :, None, None] * torch.exp(-mahal)
        alpha = torch.clamp(alpha, 0.0, 0.99)

        one_minus_alpha = 1.0 - alpha

        transmittance = torch.ones_like(alpha)
        if max_T > 1:
            transmittance[:, 1:] = torch.cumprod(one_minus_alpha[:, :-1], dim=1)

        weights = alpha * transmittance
        rendered = torch.einsum("ctk,ctpq->ckpq", g_colors, weights)

        remaining_t = transmittance[:, -1] * one_minus_alpha[:, -1]
        rendered = rendered + bg_color[None, :, None, None] * remaining_t[:, None, :, :]

        if rendered.dtype != image_tiles.dtype:
            rendered = rendered.to(image_tiles.dtype)
        image_tiles[ty, tx] = rendered

    return image_pad[:, :H, :W]


def _rasterize_gsplat(
    gaussians: GaussianParams,
    H: int,
    W: int,
    bg_color: Float[Tensor, " 3"] | None = None,
    tile_size: int = 16,
) -> Float[Tensor, "3 H W"]:
    """CUDA-accelerated rasterizer using gsplat low-level primitives.

    Requires a CUDA device and the gsplat library (``pip install gsplat``).
    Gaussians must arrive pre-sorted by depth (model handles this).
    """
    from gsplat import isect_tiles, isect_offset_encode, rasterize_to_pixels

    device = gaussians.means.device
    dtype = gaussians.means.dtype
    G = gaussians.means.shape[0]

    if bg_color is None:
        bg_color = torch.ones(3, device=device, dtype=dtype)

    if G == 0:
        return bg_color[:, None, None].expand(3, H, W).clone()

    # gsplat expects post-sigmoid opacities in [0, 1].
    opacities = torch.sigmoid(gaussians.opacities)

    # Compute conics (Σ⁻¹ upper triangle) and radii directly from scales/rotations
    # without allocating intermediate (G, 2, 2) tensors.
    cos = torch.cos(gaussians.rotations)
    sin = torch.sin(gaussians.rotations)
    sx2 = gaussians.scales[:, 0] ** 2
    sy2 = gaussians.scales[:, 1] ** 2

    # Covariance diagonal + off-diagonal (scalar components).
    cov_a = cos * cos * sx2 + sin * sin * sy2  # Σ[0,0]
    cov_b = cos * sin * (sx2 - sy2)            # Σ[0,1]
    cov_d = sin * sin * sx2 + cos * cos * sy2  # Σ[1,1]

    det = (cov_a * cov_d - cov_b * cov_b).clamp(min=1e-8)
    inv_det = 1.0 / det
    conics = torch.stack([cov_d * inv_det, -cov_b * inv_det, cov_a * inv_det], dim=-1)  # (G, 3)

    # Per-axis radii: 3σ bounding in each axis, rounded up. (G, 2) int32.
    radii = torch.stack([
        (3.0 * torch.sqrt(cov_a)).ceil(),
        (3.0 * torch.sqrt(cov_d)).ceil(),
    ], dim=-1).int()  # (G, 2)

    # Synthetic monotonic depths — Gaussians are already globally sorted by
    # the model, so monotonic indices preserve that order in gsplat's per-tile
    # radix sort.
    depths = torch.arange(G, device=device, dtype=torch.float32)

    means2d = gaussians.means.contiguous()  # (G, 2)
    colors = gaussians.colors.contiguous()  # (G, 3)

    tile_width = (W + tile_size - 1) // tile_size
    tile_height = (H + tile_size - 1) // tile_size

    # Tile intersection bookkeeping — isect_tiles needs batch dim (1, G, ...).
    _, isect_ids, flatten_ids = isect_tiles(
        means2d.unsqueeze(0),   # (1, G, 2)
        radii.unsqueeze(0),     # (1, G, 2)
        depths.unsqueeze(0),    # (1, G)
        tile_size,
        tile_width,
        tile_height,
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    # gsplat rasterize_to_pixels expects batch dims [C, ...] where C=1.
    # Parameter order is (image_width, image_height) — i.e. (W, H).
    rendered, _ = rasterize_to_pixels(
        means2d.unsqueeze(0),         # (1, G, 2)
        conics.unsqueeze(0),          # (1, G, 3)
        colors.unsqueeze(0),          # (1, G, 3)
        opacities.unsqueeze(0),       # (1, G)
        W,                            # image_width
        H,                            # image_height
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=bg_color.unsqueeze(0),  # (1, 3)
    )

    # rendered: (1, H, W, 3) → (3, H, W)
    return rendered.squeeze(0).permute(2, 0, 1)


def rasterize(
    gaussians: GaussianParams,
    H: int,
    W: int,
    bg_color: Float[Tensor, " 3"] | None = None,
    tile_size: int = 16,
    chunk_size: int = 16,
    backend: RasterBackend = "mps",
) -> Float[Tensor, "3 H W"]:
    """Differentiable 2D Gaussian splatting.

    Args:
        gaussians: GaussianParams with G total Gaussians.
        H, W: Image height and width in pixels.
        bg_color: Background color (3,). Defaults to white.
        tile_size: Tile size in pixels.
        chunk_size: Number of tiles processed in parallel.
        backend: ``"pytorch"`` (default), ``"auto"``, ``"gsplat"``,
                 or legacy aliases ``"mps"``/``"reference"``.

    Returns:
        Rendered image (3, H, W) in [0, 1].
    """
    resolved = _resolve_backend(backend, gaussians.means.device)

    if resolved == "pytorch":
        # Preserve explicit chunk override; otherwise auto-tune.
        pytorch_chunk = chunk_size
        if chunk_size == 16:
            pytorch_chunk = _default_pytorch_chunk_size(
                H,
                W,
                tile_size,
                gaussians.means.shape[0],
            )
        return _rasterize_pytorch(
            gaussians,
            H,
            W,
            bg_color=bg_color,
            tile_size=tile_size,
            chunk_size=pytorch_chunk,
        )

    if resolved == "gsplat":
        return _rasterize_gsplat(
            gaussians,
            H,
            W,
            bg_color=bg_color,
            tile_size=tile_size,
        )

    raise RuntimeError(f"Unreachable backend resolution: {resolved!r}")
