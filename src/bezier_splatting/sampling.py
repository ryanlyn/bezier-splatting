"""Gaussian parameter computation from Bézier curves.

Control points are stored in normalized [-1, 1] coordinates and scaled to
pixel coordinates at sampling time via (H, W) arguments.
"""

from dataclasses import dataclass

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


def _central_diff_angles(points: Float[Tensor, "*batch K 2"]) -> Float[Tensor, "*batch K"]:
    """Rotation angles via central differences of sampled points (paper Eq. 8).

    Args:
        points: (..., K, 2) ordered sample positions.

    Returns:
        angles: (..., K) rotation in radians.
    """
    # Interior: (X_{k+1} - X_{k-1})
    central = points[..., 2:, :] - points[..., :-2, :]  # (..., K-2, 2)
    # Boundary: forward diff at k=0, backward diff at k=K-1
    fwd = points[..., 1:2, :] - points[..., 0:1, :]     # (..., 1, 2)
    bwd = points[..., -1:, :] - points[..., -2:-1, :]    # (..., 1, 2)
    diffs = torch.cat([fwd, central, bwd], dim=-2)       # (..., K, 2)
    return torch.atan2(diffs[..., 1], diffs[..., 0])


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
            return GaussianParams(
                means=torch.empty(0, 2, device=device),
                scales=torch.empty(0, 2, device=device),
                rotations=torch.empty(0, device=device),
                colors=torch.empty(0, 3, device=device),
                opacities=torch.empty(0, device=device),
                curve_ids=torch.empty(0, dtype=torch.long, device=device),
            )

        # Scale control points to pixel coordinates
        cp_px = model_to_pixel(control_points, H, W)

        # Evaluate composite curve at K points (tangents unused — central diffs below)
        points, _ = evaluate_composite_bezier(cp_px, K, compute_tangents=False)
        # points: (N, K, 2)

        # Rotation from central differences of sampled points (paper Eq. 8)
        angles = _central_diff_angles(points)  # (N, K)

        # σ_x: distance to next sample / rho + offset to prevent zero
        diffs = torch.diff(points, dim=1)  # (N, K-1, 2)
        spacings = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-12)  # (N, K-1)
        spacings = torch.cat([spacings, spacings[:, -1:]], dim=1)  # (N, K)
        sigma_x = spacings / self.rho + 0.5  # (N, K) — offset guarantees positivity

        # σ_y: stroke width — sigmoid maps to [0.5, 5] pixel range
        sigma_y = 0.5 + torch.sigmoid(stroke_widths).unsqueeze(-1).expand(-1, K) * 4.5

        # Per-segment opacity: assign each sample to the segment it was
        # generated from.  Uses the same partition as evaluate_composite_bezier.
        seg_sizes = composite_segment_sizes(K)
        opacity_per_sample = torch.zeros(N, K, device=device)
        k = 0
        for seg, n_seg in enumerate(seg_sizes):
            opacity_per_sample[:, k:k + n_seg] = opacities[:, seg:seg + 1].expand(-1, n_seg)
            k += n_seg

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
    ):
        self.num_intermediate = num_intermediate
        self.samples_per_curve = samples_per_curve
        self.rho = rho
        self.boundary_bias = boundary_bias

    def __call__(
        self,
        boundary_cp: Float[Tensor, "N 2 CP 2"],
        colors: Float[Tensor, "N 3"],
        opacities: Float[Tensor, " N"],
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
            return GaussianParams(
                means=torch.empty(0, 2, device=device),
                scales=torch.empty(0, 2, device=device),
                rotations=torch.empty(0, device=device),
                colors=torch.empty(0, 3, device=device),
                opacities=torch.empty(0, device=device),
                curve_ids=torch.empty(0, dtype=torch.long, device=device),
            )

        # Scale CPs to pixel coordinates
        bcp_px = model_to_pixel(boundary_cp, H, W)  # (N, 2, num_cp, 2)

        top_cp = bcp_px[:, 0]   # (N, num_cp, 2)
        bot_cp = bcp_px[:, 1]   # (N, num_cp, 2)

        # Interpolation weights for R+2 curves (including boundaries at 0 and 1)
        # Use boundary-biased spacing: power-law pushes samples toward boundaries
        uniform = torch.linspace(0, 1, R_total, device=device, dtype=boundary_cp.dtype)
        # Apply boundary bias via power-law: more samples near 0 and 1
        # beta distribution approximation: 0.5 - 0.5*cos(pi*u^(1/bias))
        biased = 0.5 - 0.5 * torch.cos(torch.pi * uniform.pow(1.0 / self.boundary_bias))
        interp_weights = biased  # (R_total,)

        # Interpolate control points: P^(k) = (1 - w_k)*top + w_k*bot
        w = interp_weights[None, :, None, None]  # (1, R_total, 1, 1)
        interp_cp = (1 - w) * top_cp[:, None] + w * bot_cp[:, None]
        # (N, R_total, num_cp, 2)

        num_cp = boundary_cp.shape[2]

        # Sample K points along each curve — avoid exact endpoints where
        # tangent direction can be degenerate and scales might be zero
        t = torch.linspace(0.007, 0.993, K, device=device, dtype=boundary_cp.dtype)

        flat_cp = interp_cp.reshape(N * R_total, num_cp, 2)
        points = evaluate_bezier(flat_cp, t)    # (N*R_total, K, 2)

        points = points.reshape(N, R_total, K, 2)

        # Rotation from central differences of sampled points (paper Eq. 8)
        angles = _central_diff_angles(points)  # (N, R_total, K)

        # σ_x: along-curve spacing
        diffs_along = torch.diff(points, dim=2)  # (N, R_total, K-1, 2)
        spacings_along = torch.sqrt((diffs_along ** 2).sum(dim=-1) + 1e-12)
        spacings_along = torch.cat([spacings_along, spacings_along[..., -1:]], dim=2)
        sigma_x = spacings_along / self.rho  # (N, R_total, K)

        # σ_y: cross-curve spacing (distance to next curve at same k)
        # Paper Eq. 7: σ_y(r,k) = |X_{r+1,k} - X_{r,k}| / rho
        # At shared endpoints, boundaries pinch together → spacing goes to 0.
        diffs_cross = torch.diff(points, dim=1)  # (N, R_total-1, K, 2)
        spacings_cross = torch.sqrt((diffs_cross ** 2).sum(dim=-1) + 1e-12)  # (N, R_total-1, K)
        # Pad: last curve uses spacing to previous (not first!)
        spacings_cross = torch.cat([spacings_cross, spacings_cross[:, -1:]], dim=1)  # (N, R_total, K)
        sigma_y = spacings_cross / self.rho  # (N, R_total, K)

        # --- Mode-specific scale clamping (all non-inplace for autograd) ---

        # Boundary taper: attenuate σ_x at first/last 3 samples to soften endpoints
        if K >= 3:
            taper = torch.tensor([0.4, 0.9, 1.0], device=device, dtype=sigma_x.dtype)
            # Build taper mask for full K dimension: [0.4, 0.9, 1.0, ..., 1.0, ..., 1.0, 0.9, 0.4]
            taper_full = torch.ones(K, device=device, dtype=sigma_x.dtype)
            taper_full[:3] = taper
            taper_full[-3:] = taper.flip(0)
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
        opacities_expanded = opacities[:, None, None].expand(-1, R_total, K).reshape(G)

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
