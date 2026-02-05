"""VectorGraphicsScene — top-level differentiable scene model.

All control points are stored in [0, 1] normalized coordinates.
Scaling to pixel coordinates happens at sampling/rendering time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .rasterizer import rasterize
from .sampling import ClosedCurveSampler, GaussianParams, OpenCurveSampler


class VectorGraphicsScene(nn.Module):
    """Differentiable vector graphics scene composed of open strokes and closed fills.

    Open curves: 3 connected cubic Bézier segments (10 CPs each), rendered as stroked paths.
    Closed curves: paired boundary curves (shared endpoints) defining filled regions.

    All control points are in [0, 1] normalized coordinates.
    """

    def __init__(
        self,
        n_open: int = 128,
        n_closed: int = 64,
        H: int = 256,
        W: int = 256,
        closed_cp: int = 4,
        samples_per_open: int = 20,
        samples_per_closed_curve: int = 15,
        num_intermediate: int = 20,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.n_open = n_open
        self.n_closed = n_closed

        # ── Open curve parameters (all in [0, 1] normalized coords) ──
        if n_open > 0:
            self.open_control_points = nn.Parameter(self._init_open_cps(n_open))
            self.open_colors = nn.Parameter(torch.rand(n_open, 3))
            self.open_opacities = nn.Parameter(torch.zeros(n_open, 3))  # 3 per curve (per segment)
            self.open_stroke_widths = nn.Parameter(torch.zeros(n_open))
        else:
            self.register_buffer("open_control_points", torch.empty(0, 10, 2))
            self.register_buffer("open_colors", torch.empty(0, 3))
            self.register_buffer("open_opacities", torch.empty(0, 3))
            self.register_buffer("open_stroke_widths", torch.empty(0))

        # ── Closed curve parameters (shared endpoints enforced) ──
        if n_closed > 0:
            self.closed_boundary_cp = nn.Parameter(
                self._init_closed_cps(n_closed, closed_cp)
            )
            self.closed_colors = nn.Parameter(torch.rand(n_closed, 3))
            self.closed_opacities = nn.Parameter(torch.zeros(n_closed))
        else:
            self.register_buffer("closed_boundary_cp", torch.empty(0, 2, closed_cp, 2))
            self.register_buffer("closed_colors", torch.empty(0, 3))
            self.register_buffer("closed_opacities", torch.empty(0))

        # ── Samplers ──
        self.open_sampler = OpenCurveSampler(samples_per_curve=samples_per_open)
        self.closed_sampler = ClosedCurveSampler(
            num_intermediate=num_intermediate,
            samples_per_curve=samples_per_closed_curve,
        )

    @staticmethod
    def _init_open_cps(n: int) -> Tensor:
        """Initialize open curve CPs in [0, 1] normalized coords."""
        centers = torch.rand(n, 2)
        t = torch.linspace(-1, 1, 10).unsqueeze(0).unsqueeze(-1)
        progression = t * torch.tensor([[[0.1, 0]]])
        offsets = torch.randn(n, 10, 2) * 0.03
        cps = centers.unsqueeze(1) + progression + offsets
        return cps.clamp(0, 1)

    @staticmethod
    def _init_closed_cps(n: int, num_cp: int) -> Tensor:
        """Initialize closed curve boundary CPs with shared endpoints."""
        cps = torch.zeros(n, 2, num_cp, 2)
        centers = torch.rand(n, 2)
        size = 0.08

        for boundary in range(2):
            t = torch.linspace(0, 1, num_cp).unsqueeze(0)
            y_offset = size * (1 if boundary == 0 else -1)
            x_vals = centers[:, 0:1] + (t - 0.5) * size * 2
            y_vals = centers[:, 1:2] + y_offset + torch.randn(n, num_cp) * size * 0.3
            cps[:, boundary, :, 0] = x_vals
            cps[:, boundary, :, 1] = y_vals

        # Enforce shared endpoints: first and last CPs of both boundaries must match
        shared_start = (cps[:, 0, 0] + cps[:, 1, 0]) / 2
        shared_end = (cps[:, 0, -1] + cps[:, 1, -1]) / 2
        cps[:, 0, 0] = shared_start
        cps[:, 1, 0] = shared_start
        cps[:, 0, -1] = shared_end
        cps[:, 1, -1] = shared_end

        return cps.clamp(0, 1)

    def _enforce_shared_endpoints(self) -> Tensor:
        """Return closed boundary CPs with shared endpoints enforced.

        Averages the first and last CPs of each boundary pair so they remain
        coincident during optimization.
        """
        bcp = self.closed_boundary_cp  # (N, 2, num_cp, 2)
        if bcp.shape[0] == 0:
            return bcp

        shared_start = (bcp[:, 0, 0] + bcp[:, 1, 0]) / 2  # (N, 2)
        shared_end = (bcp[:, 0, -1] + bcp[:, 1, -1]) / 2

        # Build new tensor with shared endpoints
        bcp_fixed = bcp.clone()
        bcp_fixed[:, 0, 0] = shared_start
        bcp_fixed[:, 1, 0] = shared_start
        bcp_fixed[:, 0, -1] = shared_end
        bcp_fixed[:, 1, -1] = shared_end
        return bcp_fixed

    def forward(self, H: int | None = None, W: int | None = None) -> Tensor:
        """Render the scene."""
        H = H or self.H
        W = W or self.W

        all_gaussians: list[GaussianParams] = []
        curve_areas: list[tuple[Tensor, int]] = []  # (area_per_curve, n_gaussians_per_curve)
        curve_id_offset = 0

        # ── Sample from open curves ──
        if self.n_open > 0:
            open_g = self.open_sampler(
                self.open_control_points,
                torch.sigmoid(self.open_colors),
                self.open_opacities,
                self.open_stroke_widths,
                H, W,
                curve_id_offset=curve_id_offset,
            )
            if open_g.means.shape[0] > 0:
                all_gaussians.append(open_g)
                # Per-curve area: compute from CPs scaled to pixels
                cp_px = self.open_control_points * torch.tensor([W, H], device=self.open_control_points.device, dtype=self.open_control_points.dtype)
                edge_len = torch.norm(cp_px[:, 1:] - cp_px[:, :-1], dim=-1).sum(dim=-1)  # (N,)
                sw = 0.5 + torch.sigmoid(self.open_stroke_widths) * 4.5
                areas = edge_len * sw  # (N,)
                K = self.open_sampler.samples_per_curve
                curve_areas.append((areas, K))
            curve_id_offset += self.n_open

        # ── Sample from closed curves ──
        if self.n_closed > 0:
            bcp = self._enforce_shared_endpoints()
            closed_g = self.closed_sampler(
                bcp,
                torch.sigmoid(self.closed_colors),
                self.closed_opacities,
                H, W,
                curve_id_offset=curve_id_offset,
            )
            if closed_g.means.shape[0] > 0:
                all_gaussians.append(closed_g)
                # Per-curve area from bounding box of CPs
                bcp_px = bcp * torch.tensor([W, H], device=bcp.device, dtype=bcp.dtype)
                cp_flat = bcp_px.reshape(bcp_px.shape[0], -1, 2)
                bb = cp_flat.max(dim=1).values - cp_flat.min(dim=1).values
                areas = bb[:, 0] * bb[:, 1]  # (N,)
                R_total = self.closed_sampler.num_intermediate + 2
                K = self.closed_sampler.samples_per_curve
                curve_areas.append((areas, R_total * K))

        if len(all_gaussians) == 0:
            device = self.open_control_points.device if self.n_open > 0 else self.closed_boundary_cp.device
            return torch.ones(3, H, W, device=device)

        # Concatenate all Gaussians
        if len(all_gaussians) == 1:
            combined = all_gaussians[0]
        else:
            combined = all_gaussians[0]
            for g in all_gaussians[1:]:
                combined = combined.concat(g)

        # ── Per-curve depth sort (paper Sec 3.3) ──
        # All Gaussians from the same curve share the same depth.
        # Smallest-area curves are frontmost (index 0 in front-to-back compositing).
        all_areas_list = []
        for areas, gaussians_per_curve in curve_areas:
            # Expand curve area to each Gaussian from that curve
            expanded = areas.unsqueeze(1).expand(-1, gaussians_per_curve).reshape(-1)
            all_areas_list.append(expanded)
        all_areas = torch.cat(all_areas_list, dim=0)

        sort_indices = torch.argsort(all_areas, descending=False)
        combined = GaussianParams(
            means=combined.means[sort_indices],
            scales=combined.scales[sort_indices],
            rotations=combined.rotations[sort_indices],
            colors=combined.colors[sort_indices],
            opacities=combined.opacities[sort_indices],
            curve_ids=combined.curve_ids[sort_indices],
        )

        return rasterize(combined, H, W)
