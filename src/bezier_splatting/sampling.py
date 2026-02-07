"""Gaussian parameter computation from Bézier curves.

Control points are stored in normalized [-1, 1] coordinates and scaled to
pixel coordinates at sampling time via (H, W) arguments.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from .bezier import evaluate_bezier, evaluate_composite_bezier, composite_segment_sizes
from .coords import model_to_pixel


@dataclass
class GaussianParams:
    """Parameters for a set of 2D Gaussians ready for rasterization."""

    means: Float[Tensor, "G 2"]       # center positions in pixel coords
    scales: Float[Tensor, "G 2"]      # [σ_x, σ_y] standard deviations
    rotations: Float[Tensor, " G"]    # rotation angle θ in radians
    colors: Float[Tensor, "G 3"]      # RGB in [0, 1]
    opacities: Float[Tensor, " G"]    # pre-sigmoid opacity
    curve_ids: Int[Tensor, " G"]      # which curve each Gaussian belongs to (for per-curve depth)

    def concat(self, other: "GaussianParams") -> "GaussianParams":
        """Concatenate two GaussianParams along the Gaussian dimension."""
        return GaussianParams(
            means=torch.cat([self.means, other.means], dim=0),
            scales=torch.cat([self.scales, other.scales], dim=0),
            rotations=torch.cat([self.rotations, other.rotations], dim=0),
            colors=torch.cat([self.colors, other.colors], dim=0),
            opacities=torch.cat([self.opacities, other.opacities], dim=0),
            curve_ids=torch.cat([self.curve_ids, other.curve_ids], dim=0),
        )


def _empty_gaussian_params(device: torch.device) -> GaussianParams:
    """Return an empty GaussianParams container on ``device``."""
    return GaussianParams(
        means=torch.empty(0, 2, device=device),
        scales=torch.empty(0, 2, device=device),
        rotations=torch.empty(0, device=device),
        colors=torch.empty(0, 3, device=device),
        opacities=torch.empty(0, device=device),
        curve_ids=torch.empty(0, dtype=torch.long, device=device),
    )


def _safe_pairwise_spacing(points: Tensor, sample_dim: int) -> Tensor:
    """Compute pairwise Euclidean spacing with terminal replication."""
    diffs = torch.diff(points, dim=sample_dim)
    spacing = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-12)

    tail_idx = [slice(None)] * spacing.ndim
    tail_idx[sample_dim] = slice(-1, None)
    tail = spacing[tuple(tail_idx)]
    return torch.cat([spacing, tail], dim=sample_dim)


def _closed_spacing_permutation(rows_total: int, device: torch.device) -> Tensor:
    """Permutation from output row order to spacing row order.

    Output order is ``[top, bottom, interiors...]`` while spacing expects
    ``[top, interiors..., bottom]``.
    """
    if rows_total <= 2:
        return torch.arange(rows_total, device=device, dtype=torch.long)

    perm = torch.empty(rows_total, device=device, dtype=torch.long)
    perm[0] = 0
    perm[1:-1] = torch.arange(2, rows_total, device=device, dtype=torch.long)
    perm[-1] = 1
    return perm


def _central_diff_angles(points: Float[Tensor, "*batch K 2"]) -> Float[Tensor, "*batch K"]:
    """Rotation angles via central differences of sampled points (paper Eq. 8).

    Args:
        points: (..., K, 2) ordered sample positions.

    Returns:
        angles: (..., K) rotation in radians.
    """
    # Replicate-pad K dim by 1 on each side so central diffs naturally
    # become forward diff at k=0 and backward diff at k=K-1.
    padded = F.pad(points, (0, 0, 1, 1), mode="replicate")  # (..., K+2, 2)
    diffs = padded[..., 2:, :] - padded[..., :-2, :]        # (..., K, 2)
    return torch.atan2(diffs[..., 1], diffs[..., 0])


def _closed_interior_weights(
    num_intermediate: int,
    mode: str,
    boundary_bias: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Float[Tensor, " R"]:
    """Interpolation weights for interior closed-curve rows.

    Returns weights in (0, 1) for ``R=num_intermediate`` interior rows.
    Boundaries are handled separately as rows 0 (top, w=0) and 1 (bottom, w=1).
    """
    if num_intermediate == 0:
        return torch.empty(0, dtype=dtype, device=device)

    if mode == "boundary_biased":
        # Legacy mode: cosine + power-law bias.
        r_total = num_intermediate + 2
        uniform = torch.linspace(0, 1, r_total, device=device, dtype=dtype)
        biased = 0.5 - 0.5 * torch.cos(torch.pi * uniform.pow(1.0 / boundary_bias))
        return biased[1:-1]

    if mode == "cdf":
        # Official-style mode: Normal CDF over [-2, 2], no explicit endpoint rows.
        grid = torch.linspace(-2, 2, num_intermediate, device=device, dtype=dtype)
        return 0.5 * (1.0 + torch.erf(grid / math.sqrt(2.0) / 0.85))

    raise ValueError(
        f"Unknown sampling mode {mode!r}. Expected 'boundary_biased' or 'cdf'.",
    )


def _expand_closed_opacities(
    opacities: Tensor,
    rows_total: int,
    samples_per_curve: int,
) -> Tensor:
    """Expand per-curve closed opacity params to per-Gaussian opacities.

    Supports:
        - legacy scalar: (N,) or (N, 1)
        - profile: (N, 3) = [boundary_top, interior_mid, boundary_bottom]
    """
    if opacities.ndim == 1:
        profile = opacities[:, None].expand(-1, 3)
    elif opacities.ndim == 2 and opacities.shape[1] == 1:
        profile = opacities.expand(-1, 3)
    elif opacities.ndim == 2 and opacities.shape[1] >= 3:
        profile = opacities[:, :3]
    else:
        raise ValueError(
            f"closed opacities must have shape (N,), (N,1), or (N,3+). Got {tuple(opacities.shape)}.",
        )

    top = profile[:, 0:1]
    middle = profile[:, 1:2]
    bottom = profile[:, 2:3]

    num_intermediate = rows_total - 2
    if num_intermediate > 0:
        split = num_intermediate // 2
        first_len = split
        second_len = num_intermediate - split

        parts: list[Tensor] = []
        if first_len > 0:
            w1 = torch.linspace(0, 1, first_len, device=profile.device, dtype=profile.dtype).unsqueeze(0)
            parts.append((1 - w1) * top + w1 * middle)
        if second_len > 0:
            w2 = torch.linspace(0, 1, second_len, device=profile.device, dtype=profile.dtype).unsqueeze(0)
            parts.append((1 - w2) * middle + w2 * bottom)
        area = torch.cat(parts, dim=1) if parts else torch.empty(profile.shape[0], 0, device=profile.device, dtype=profile.dtype)
    else:
        area = torch.empty(profile.shape[0], 0, device=profile.device, dtype=profile.dtype)

    row_opacity = torch.cat([top, bottom, area], dim=1)  # (N, rows_total)
    return row_opacity.unsqueeze(-1).expand(-1, -1, samples_per_curve).reshape(-1)


class OpenCurveSampler:
    """Sample K Gaussians along each open Bézier curve (3 connected cubics).

    Each open curve has:
        - 10 control points in [-1, 1] normalized coords
        - 1 color (RGB)
        - 3 opacities (one per cubic segment, paper Appendix D)
        - 1 stroke width (scalar, learnable)

    Control points are scaled to pixel coords at sampling time.
    """

    def __init__(self, samples_per_curve: int = 20, rho: float = 2.0):
        self.samples_per_curve = samples_per_curve
        self.rho = rho

    def __call__(
        self,
        control_points: Float[Tensor, "N 10 2"],
        colors: Float[Tensor, "N 3"],
        opacities: Float[Tensor, "N 3"],
        stroke_widths: Float[Tensor, " N"],
        H: int = 256,
        W: int = 256,
        curve_id_offset: int = 0,
    ) -> GaussianParams:
        N = control_points.shape[0]
        K = self.samples_per_curve
        device = control_points.device

        if N == 0:
            return _empty_gaussian_params(device)

        # Scale control points to pixel coordinates
        cp_px = model_to_pixel(control_points, H, W)

        # Evaluate composite curve at K points (tangents unused — central diffs below)
        points, _ = evaluate_composite_bezier(cp_px, K, compute_tangents=False)
        # points: (N, K, 2)

        # Rotation from central differences of sampled points (paper Eq. 8)
        angles = _central_diff_angles(points)  # (N, K)

        # σ_x: distance to next sample / rho + offset to prevent zero
        spacings = _safe_pairwise_spacing(points, sample_dim=1)  # (N, K)
        sigma_x = spacings / self.rho + 0.5  # (N, K) — offset guarantees positivity

        # σ_y: stroke width — sigmoid maps to [0.5, 5] pixel range
        sigma_y = 0.5 + torch.sigmoid(stroke_widths).unsqueeze(-1).expand(-1, K) * 4.5

        # Per-segment opacity: assign each sample to the segment it was
        # generated from.  Uses the same partition as evaluate_composite_bezier.
        seg_sizes = composite_segment_sizes(K)
        seg_sizes_t = torch.tensor(seg_sizes, device=device)
        opacity_per_sample = torch.repeat_interleave(opacities, seg_sizes_t, dim=1)

        # Flatten (N, K) → (N*K,)
        means = points.reshape(-1, 2)
        scales_out = torch.stack([sigma_x, sigma_y], dim=-1).reshape(-1, 2)
        rotations_out = angles.reshape(-1)
        colors_expanded = colors.unsqueeze(1).expand(-1, K, -1).reshape(-1, 3)
        opacities_expanded = opacity_per_sample.reshape(-1)

        # Curve IDs: all K Gaussians from curve i share the same ID
        ids = torch.arange(N, device=device, dtype=torch.long).unsqueeze(1).expand(-1, K).reshape(-1)
        ids = ids + curve_id_offset

        return GaussianParams(
            means=means,
            scales=scales_out,
            rotations=rotations_out,
            colors=colors_expanded,
            opacities=opacities_expanded,
            curve_ids=ids,
        )


class ClosedCurveSampler:
    """Fill a region between two boundary curves with Gaussians.

    Paper Eq. 4-6: R intermediate curves + 2 boundaries = R+2 total.
    Boundary curves must share start and end points.

    Control points are in [-1, 1] normalized coords, scaled at sample time.
    """

    def __init__(
        self,
        num_intermediate: int = 20,
        samples_per_curve: int = 15,
        rho: float = 2.0,
        boundary_bias: float = 2.0,
        sampling_mode: str = "boundary_biased",
    ):
        self.num_intermediate = num_intermediate
        self.samples_per_curve = samples_per_curve
        self.rho = rho
        self.boundary_bias = boundary_bias
        self.sampling_mode = sampling_mode
        # Pre-compute interior weights (deterministic, depends only on R/mode/bias)
        self._interior_weights_cpu = _closed_interior_weights(
            num_intermediate, sampling_mode, boundary_bias,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        self._interior_weights_cache: Tensor | None = None

    def __call__(
        self,
        boundary_cp: Float[Tensor, "N 2 CP 2"],
        colors: Float[Tensor, "N 3"],
        opacities: Float[Tensor, "N 3"] | Float[Tensor, " N"],
        H: int = 256,
        W: int = 256,
        curve_id_offset: int = 0,
    ) -> GaussianParams:
        N = boundary_cp.shape[0]
        R = self.num_intermediate
        R_total = R + 2  # R intermediate + 2 boundaries (paper Eq. 6)
        K = self.samples_per_curve
        device = boundary_cp.device

        if N == 0:
            return _empty_gaussian_params(device)

        # Scale CPs to pixel coordinates
        bcp_px = model_to_pixel(boundary_cp, H, W)  # (N, 2, num_cp, 2)

        top_cp = bcp_px[:, 0]   # (N, num_cp, 2)
        bot_cp = bcp_px[:, 1]   # (N, num_cp, 2)

        # Build row-ordered control points:
        #   row 0 -> top boundary
        #   row 1 -> bottom boundary
        #   rows 2.. -> interior interpolated curves
        # Use pre-computed weights, moving to correct device/dtype on first use
        cache = self._interior_weights_cache
        if cache is None or cache.device != device or cache.dtype != boundary_cp.dtype:
            self._interior_weights_cache = self._interior_weights_cpu.to(
                device=device, dtype=boundary_cp.dtype,
            )
            cache = self._interior_weights_cache
        interior_w = cache  # (R,)

        if R > 0:
            w = interior_w[None, :, None, None]  # (1, R, 1, 1)
            interior_cp = (1 - w) * top_cp[:, None] + w * bot_cp[:, None]  # (N, R, CP, 2)
            row_cp = torch.cat([top_cp[:, None], bot_cp[:, None], interior_cp], dim=1)
        else:
            row_cp = torch.cat([top_cp[:, None], bot_cp[:, None]], dim=1)
        # row_cp: (N, R_total, CP, 2)

        num_cp = boundary_cp.shape[2]

        # Sample K points along each curve — avoid exact endpoints where
        # tangent direction can be degenerate and scales might be zero
        t = torch.linspace(0.007, 0.993, K, device=device, dtype=boundary_cp.dtype)

        flat_cp = row_cp.reshape(N * R_total, num_cp, 2)
        points = evaluate_bezier(flat_cp, t)    # (N*R_total, K, 2)

        points = points.reshape(N, R_total, K, 2)

        # Rotation from central differences of sampled points (paper Eq. 8)
        angles = _central_diff_angles(points)  # (N, R_total, K)

        # σ_x: along-curve spacing
        spacings_along = _safe_pairwise_spacing(points, sample_dim=2)
        sigma_x = spacings_along / self.rho  # (N, R_total, K)

        # σ_y: cross-curve spacing (distance between adjacent rows at same k).
        # Use order [top, interiors..., bottom] for spacing computation, then
        # map back to output row order [top, bottom, interiors...].
        perm = _closed_spacing_permutation(R_total, device)
        inv_perm = torch.argsort(perm)

        points_reordered = points[:, perm]  # (N, R_total, K, 2)
        diffs_cross = torch.diff(points_reordered, dim=1)  # (N, R_total-1, K, 2)
        spacings_cross = torch.sqrt((diffs_cross ** 2).sum(dim=-1) + 1e-12)  # (N, R_total-1, K)
        sigma_y_reordered = torch.cat([spacings_cross[:, :1], spacings_cross], dim=1) / self.rho
        sigma_y = sigma_y_reordered[:, inv_perm]  # (N, R_total, K)

        # --- Mode-specific scale clamping (all non-inplace for autograd) ---

        # Boundary taper: attenuate σ_x at first/last 3 samples to soften endpoints
        if K >= 3:
            # Build taper mask for full K dimension: [0.4, 0.9, 1.0, ..., 1.0, ..., 1.0, 0.9, 0.4]
            taper_full = torch.ones(K, device=device, dtype=sigma_x.dtype)
            taper_full[0] = 0.4
            taper_full[1] = 0.9
            taper_full[-2] = 0.9
            taper_full[-1] = 0.4
            sigma_x = sigma_x * taper_full

        # Split into boundary (indices 0,1) and interior (indices 2+)
        sx_bnd = sigma_x[:, :2, :].clamp(min=0.3)
        sy_bnd = sigma_y[:, :2, :].clamp(min=0.75, max=1.0)

        if R_total > 2:
            sx_int = sigma_x[:, 2:, :]
            sy_int = sigma_y[:, 2:, :]

            # Ratio mutual clamp: when σ_y is very small, clamp both to 3:1 ratio
            threshold = 0.1
            ratio = 3.0
            mask = sy_int < threshold
            sx_int = torch.where(mask, torch.min(sx_int, sy_int * ratio), sx_int)
            sy_int = torch.where(mask, torch.min(sy_int, sx_int * ratio), sy_int)

            # Safety floor for interior curves
            sx_int = sx_int.clamp(min=0.1)
            sy_int = sy_int.clamp(min=0.1)

            sigma_x = torch.cat([sx_bnd, sx_int], dim=1)
            sigma_y = torch.cat([sy_bnd, sy_int], dim=1)
        else:
            sigma_x = sx_bnd
            sigma_y = sy_bnd

        # Flatten: (N, R_total, K) → (N*R_total*K,)
        G = N * R_total * K
        means = points.reshape(G, 2)
        scales_out = torch.stack([sigma_x, sigma_y], dim=-1).reshape(G, 2)
        rotations_flat = angles.reshape(G)

        colors_expanded = colors[:, None, None, :].expand(-1, R_total, K, -1).reshape(G, 3)
        opacities_expanded = _expand_closed_opacities(opacities, R_total, K)

        # Curve IDs: all Gaussians from closed curve i share the same ID
        ids = torch.arange(N, device=device, dtype=torch.long)[:, None, None].expand(-1, R_total, K).reshape(G)
        ids = ids + curve_id_offset

        return GaussianParams(
            means=means,
            scales=scales_out,
            rotations=rotations_flat,
            colors=colors_expanded,
            opacities=opacities_expanded,
            curve_ids=ids,
        )
