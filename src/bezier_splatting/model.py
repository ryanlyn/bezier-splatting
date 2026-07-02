"""VectorGraphicsScene — top-level differentiable scene model.

All control points are stored in [-1, 1] normalized coordinates.
Scaling to pixel coordinates happens at sampling/rendering time.
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .coords import model_to_pixel
from .rasterizer import RasterBackend, rasterize
from .sampling import ClosedCurveSampler, GaussianParams, OpenCurveSampler


class VectorGraphicsScene(nn.Module):
    """Differentiable vector graphics scene composed of open strokes and closed fills.

    Open curves: 3 connected cubic Bézier segments (10 CPs each), rendered as stroked paths.
    Closed curves: paired boundary curves (shared endpoints) defining filled regions.

    All control points are in [-1, 1] normalized coordinates.
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
        closed_sampling_mode: str = "cdf",
        raster_backend: RasterBackend = "pytorch",
        raster_tile_size: int = 16,
        raster_chunk_size: int | None = None,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.n_open = n_open
        self.n_closed = n_closed
        self.raster_backend = raster_backend
        self.raster_tile_size = raster_tile_size
        self.raster_chunk_size = raster_chunk_size

        # ── Open curve parameters (all in [-1, 1] normalized coords) ──
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

        # ── Closed curve parameters (structurally shared endpoints) ──
        if n_closed > 0:
            shared_pts, interior_cp = self._init_closed_cps(n_closed, closed_cp)
            self.closed_shared_pts = nn.Parameter(shared_pts)
            self.closed_interior_cp = nn.Parameter(interior_cp)
            self.closed_colors = nn.Parameter(torch.rand(n_closed, 3))
            # [top-boundary, interior-mid, bottom-boundary] opacity profile
            self.closed_opacities = nn.Parameter(torch.zeros(n_closed, 3))
        else:
            self.register_buffer("closed_shared_pts", torch.empty(0, 2, 2))
            self.register_buffer("closed_interior_cp", torch.empty(0, 2, closed_cp - 2, 2))
            self.register_buffer("closed_colors", torch.empty(0, 3))
            self.register_buffer("closed_opacities", torch.empty(0, 3))

        # ── Learned depth parameter (open curves first, then closed) ──
        total_curves = n_open + n_closed
        if total_curves > 0:
            self._depth = nn.Parameter(torch.ones(total_curves, 1))
        else:
            self.register_buffer("_depth", torch.zeros(0, 1))

        # ── Samplers ──
        self.open_sampler = OpenCurveSampler(samples_per_curve=samples_per_open)
        self.closed_sampler = ClosedCurveSampler(
            num_intermediate=num_intermediate,
            samples_per_curve=samples_per_closed_curve,
            sampling_mode=closed_sampling_mode,
        )
        self.closed_sampling_mode = closed_sampling_mode

    @staticmethod
    def _init_open_cps(n: int) -> Float[Tensor, "N 10 2"]:
        """Initialize open curve CPs in [-1, 1] normalized coords."""
        centers = torch.rand(n, 2) * 2 - 1  # [-1, 1]
        t = torch.linspace(-1, 1, 10).unsqueeze(0).unsqueeze(-1)
        progression = t * torch.tensor([[[0.1, 0]]])
        offsets = torch.randn(n, 10, 2) * 0.03
        cps = centers.unsqueeze(1) + progression + offsets
        return cps.clamp(-1, 1)

    @staticmethod
    def _init_closed_cps(n: int, num_cp: int) -> tuple[Float[Tensor, "N 2 2"], Float[Tensor, "N 2 interior 2"]]:
        """Initialize closed curve boundary CPs with shared endpoints.

        Returns:
            shared_pts: (N, 2, 2) — shared [start, end] × [x, y]
            interior_cp: (N, 2, num_cp-2, 2) — interior CPs per boundary
        """
        cps = torch.zeros(n, 2, num_cp, 2)
        centers = torch.rand(n, 2) * 2 - 1  # [-1, 1]
        size = 0.08

        for boundary in range(2):
            t = torch.linspace(0, 1, num_cp).unsqueeze(0)
            y_offset = size * (1 if boundary == 0 else -1)
            x_vals = centers[:, 0:1] + (t - 0.5) * size * 2
            y_vals = centers[:, 1:2] + y_offset + torch.randn(n, num_cp) * size * 0.3
            cps[:, boundary, :, 0] = x_vals
            cps[:, boundary, :, 1] = y_vals

        # Shared endpoints: average first and last CPs of both boundaries
        shared_start = (cps[:, 0, 0] + cps[:, 1, 0]) / 2
        shared_end = (cps[:, 0, -1] + cps[:, 1, -1]) / 2
        shared_pts = torch.stack([shared_start, shared_end], dim=1).clamp(-1, 1)  # (N, 2, 2)

        interior_cp = cps[:, :, 1:-1, :].clamp(-1, 1)  # (N, 2, num_cp-2, 2)

        return shared_pts, interior_cp

    def _assemble_boundary_cp(self) -> Float[Tensor, "N 2 CP 2"]:
        """Reconstruct full (N, 2, num_cp, 2) from shared endpoints + interior CPs.

        Both boundaries share the exact same start/end points because they
        are read from the single closed_shared_pts tensor.
        """
        if self.n_closed == 0:
            num_cp = self.closed_interior_cp.shape[2] + 2
            return torch.empty(0, 2, num_cp, 2, device=self.closed_interior_cp.device)

        # Broadcast shared points to both boundaries: (N, 2) -> (N, 2, 1, 2)
        start = self.closed_shared_pts[:, 0, :].unsqueeze(1).unsqueeze(1).expand(-1, 2, 1, -1)
        end = self.closed_shared_pts[:, 1, :].unsqueeze(1).unsqueeze(1).expand(-1, 2, 1, -1)
        return torch.cat([start, self.closed_interior_cp, end], dim=2)

    @property
    def closed_boundary_cp(self) -> Float[Tensor, "N 2 CP 2"]:
        """Assembled (N, 2, num_cp, 2) boundary CPs with shared endpoints.

        Read-only view for backward compatibility. For parameter access,
        use closed_shared_pts and closed_interior_cp directly.
        """
        return self._assemble_boundary_cp()

    @property
    def get_depth(self) -> Float[Tensor, "N 1"]:
        """Per-curve depth values used for sorting (smaller = rendered in front).

        Values are heuristic image-coverage fractions written by
        ``update_depth_heuristic()`` — approximately (curve footprint) / (H*W)
        for both curve types, so open and closed curves sort on a common scale.
        """
        return self._depth

    def _open_depth_values(self, H: int, W: int) -> Float[Tensor, "N 1"]:
        """Open-curve depth heuristic: polyline length * stroke width / (H*W)."""
        cp_px = model_to_pixel(self.open_control_points, H, W)
        diffs = cp_px[:, 1:] - cp_px[:, :-1]
        lengths = torch.sqrt((diffs ** 2).sum(-1) + 1e-12).sum(-1, keepdim=True)
        sw = 0.5 + torch.sigmoid(self.open_stroke_widths).unsqueeze(-1) * 4.5
        return lengths * sw / float(H * W)

    def _closed_depth_values(self, H: int, W: int) -> Float[Tensor, "N 1"]:
        """Closed-curve depth heuristic: AABB pixel area / (H*W)."""
        bcp = self._assemble_boundary_cp()
        all_pts = bcp.reshape(self.n_closed, -1, 2)  # (N_closed, 2*CP, 2)
        x_min = all_pts[..., 0].min(dim=-1).values
        x_max = all_pts[..., 0].max(dim=-1).values
        y_min = all_pts[..., 1].min(dim=-1).values
        y_max = all_pts[..., 1].max(dim=-1).values
        widths = (x_max - x_min) * (0.5 * W)
        heights = (y_max - y_min) * (0.5 * H)
        return (widths * heights / float(H * W)).unsqueeze(-1)

    def update_depth_heuristic(
        self, H: int, W: int, *, update_open: bool = True, update_closed: bool = True,
    ) -> None:
        """Overwrite depth parameter with geometric heuristics.

        Both heuristics are normalized by the image area (H*W) so open and
        closed curves share one comparable scale — a fraction of the image the
        curve roughly covers. Smaller values sort to the front.

        Called from the training loop on a schedule. The caller controls which
        curve types to update via keyword flags:
        - Closed curves: AABB area fraction (typically every forward pass)
        - Open curves: polyline length * stroke width / (H*W) (typically every
          20 steps for the first 10k iterations, then frozen; also refreshed
          right after topology changes)

        This is separated from ``get_gaussians()`` so that torch.compile can
        trace the forward path without recompilation from iteration guards.
        """
        n_open = self.n_open
        n_closed = self.n_closed

        # Closed curves: overwrite with normalized AABB area
        if update_closed and n_closed > 0:
            with torch.no_grad():
                self._depth.data[n_open:] = self._closed_depth_values(H, W)

        # Open curves: overwrite with normalized polyline length * stroke width
        if update_open and n_open > 0:
            with torch.no_grad():
                self._depth.data[:n_open] = self._open_depth_values(H, W)

    def get_gaussians(self, H: int, W: int) -> GaussianParams | None:
        """Assemble and depth-sort all Gaussians without rasterizing.

        Returns ``None`` when the scene has no curves (both n_open and
        n_closed are zero or all samplers produce empty output).

        Depth overwrites are handled by ``update_depth_heuristic()``, which
        must be called externally before this method (e.g. from the training
        loop) on the appropriate schedule.
        """
        n_open = self.n_open
        n_closed = self.n_closed
        all_gaussians: list[GaussianParams] = []
        curve_id_offset = 0
        open_samples_per_curve = 0
        closed_samples_per_curve = 0

        # ── Sample from open curves ──
        if n_open > 0:
            # Straight-through clamp: forward pass renders clamped colors, but
            # gradients pass through unclamped so out-of-range parameters keep
            # receiving a restoring signal instead of going dead.
            oc = self.open_colors
            open_colors_st = oc + (oc.clamp(0.0, 1.0) - oc).detach()
            open_g = self.open_sampler(
                self.open_control_points,
                open_colors_st,
                self.open_opacities,
                self.open_stroke_widths,
                H, W,
                curve_id_offset=curve_id_offset,
            )
            if open_g.means.shape[0] > 0:
                all_gaussians.append(open_g)
                open_samples_per_curve = self.open_sampler.samples_per_curve
            curve_id_offset += n_open

        # ── Sample from closed curves ──
        if n_closed > 0:
            bcp = self._assemble_boundary_cp()
            closed_g = self.closed_sampler(
                bcp,
                torch.sigmoid(self.closed_colors),
                self.closed_opacities,
                H, W,
                curve_id_offset=curve_id_offset,
            )
            if closed_g.means.shape[0] > 0:
                all_gaussians.append(closed_g)
                R_total = self.closed_sampler.num_intermediate + 2
                closed_samples_per_curve = R_total * self.closed_sampler.samples_per_curve

        if not all_gaussians:
            return None

        # Concatenate all Gaussians
        combined = all_gaussians[0]
        for g in all_gaussians[1:]:
            combined = combined.concat(g)

        # ── Depth-based sorting ──
        # Depth values are heuristic coverage fractions (smaller = front)
        depth_values = self.get_depth  # (total_curves, 1)

        # Build per-Gaussian depth
        depth_parts: list[Tensor] = []
        if n_open > 0 and open_samples_per_curve > 0:
            # Open: detach depth from computation graph
            open_depth_vals = depth_values[:n_open].detach()  # (n_open, 1)
            open_depth_expanded = open_depth_vals.expand(-1, open_samples_per_curve).reshape(-1)
            depth_parts.append(open_depth_expanded)

        if n_closed > 0 and closed_samples_per_curve > 0:
            # Closed: keep gradient flow (though effectively no grad since overwritten in no_grad)
            closed_depth_vals = depth_values[n_open:]  # (n_closed, 1)

            # Boundary Gaussians get offset -1e-6 so they render in front of fill
            R_total = self.closed_sampler.num_intermediate + 2
            K = self.closed_sampler.samples_per_curve
            # Per closed curve: R_total * K Gaussians, first K and second K are boundaries
            closed_depth_per_curve = closed_depth_vals.expand(-1, R_total * K)  # (n_closed, R_total*K)
            # Create offset: boundaries at rows 0 and 1 (first 2*K samples) get -1e-6
            boundary_offset = torch.zeros(R_total * K, device=depth_values.device)
            boundary_offset[:2 * K] = -1e-6
            closed_depth_per_curve = closed_depth_per_curve + boundary_offset.unsqueeze(0)
            depth_parts.append(closed_depth_per_curve.reshape(-1))

        all_depths = torch.cat(depth_parts, dim=0)
        # Stable sort so tied depths keep a deterministic (insertion) order
        sort_indices = torch.argsort(all_depths, descending=False, stable=True)

        # ── Fused sort: pack all differentiable fields into one tensor ──
        # means(2) + scales(2) + rotations(1) + colors(3) + opacities(1) = 9
        packed = torch.cat([
            combined.means,
            combined.scales,
            combined.rotations.unsqueeze(-1),
            combined.colors,
            combined.opacities.unsqueeze(-1),
        ], dim=-1)  # (G, 9)
        packed_sorted = packed[sort_indices]  # ONE IndexBackward0

        sorted_means = packed_sorted[:, :2]
        sorted_scales = packed_sorted[:, 2:4]
        sorted_rotations = packed_sorted[:, 4]
        sorted_colors = packed_sorted[:, 5:8]
        sorted_opacities = packed_sorted[:, 8]
        # curve_ids has no grad — index separately
        sorted_curve_ids = combined.curve_ids[sort_indices]

        return GaussianParams(
            means=sorted_means,
            scales=sorted_scales,
            rotations=sorted_rotations,
            colors=sorted_colors,
            opacities=sorted_opacities,
            curve_ids=sorted_curve_ids,
        )

    def forward(
        self,
        H: int | None = None,
        W: int | None = None,
        backend: RasterBackend | None = None,
        tile_size: int | None = None,
        chunk_size: int | None = None,
    ) -> Float[Tensor, "3 H W"]:
        """Render the scene."""
        H = H or self.H
        W = W or self.W

        gaussians = self.get_gaussians(H, W)
        if gaussians is None:
            device = self.open_control_points.device if self.n_open > 0 else self.closed_shared_pts.device
            return torch.ones(3, H, W, device=device)

        return rasterize(
            gaussians,
            H,
            W,
            tile_size=self.raster_tile_size if tile_size is None else tile_size,
            chunk_size=self.raster_chunk_size if chunk_size is None else chunk_size,
            backend=self.raster_backend if backend is None else backend,
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict with backward compatibility for legacy closed opacity shape.

        Legacy checkpoints stored ``closed_opacities`` as shape ``(N,)``.
        Current model expects ``(N, 3)`` profile logits.
        """
        if "closed_opacities" in state_dict:
            val = state_dict["closed_opacities"]
            if isinstance(val, Tensor) and (val.ndim == 1 or (val.ndim == 2 and val.shape[1] == 1)):
                upgraded = val.reshape(val.shape[0], 1).expand(-1, 3).clone()
                state_dict = dict(state_dict)
                state_dict["closed_opacities"] = upgraded
        return super().load_state_dict(state_dict, strict=strict)
