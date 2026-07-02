"""Pruning and densification heuristics for topology management.

Computes masks and centers for pruning low-quality curves and densifying
at high-error regions. All logic here is pure computation — no in-place
scene mutation. The caller (optimization.py) applies the masks.

Control points are in [-1, 1] model space.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from .area import closed_curve_enclosed_area
from .coords import model_to_pixel
from .model import VectorGraphicsScene


@dataclass
class PruneConfig:
    """Configuration for pruning heuristics."""

    # Outside-image pruning
    outside_ratio_threshold: float = 0.6  # prune if >60% of AABB is outside [-1,1]

    # Overlap + color similarity suppression
    iou_threshold_open: float = 0.2
    iou_threshold_closed: float = 0.1
    color_threshold_open: float = 0.03
    color_threshold_closed: float = 0.05
    weighted_iou_threshold: float = 0.5

    # Tiny curve removal (pixel space)
    tiny_width_open: float = 4.0
    tiny_height_open: float = 4.0
    tiny_area_open: float = 12.0
    tiny_width_closed: float = 5.0
    tiny_height_closed: float = 5.0
    tiny_area_closed: float = 16.0

    # Opacity thresholds (staged)
    opacity_threshold_open: float = 0.6
    opacity_threshold_closed_early: float = 0.1  # before 70% progress
    opacity_threshold_closed_late: float = 0.3  # after 70% progress

    # Area thresholds (staged, in pixel^2 units)
    area_threshold_early: float = 50000.0  # before 60% progress
    area_threshold_mid: float = 20000.0  # 60-80% progress
    area_threshold_late: float = 2000.0  # after 80% progress


# ── Geometric helpers ────────────────────────────────────────────────────


def compute_aabb(
    control_points: Float[Tensor, "N CP 2"], H: int, W: int,
) -> Float[Tensor, "N 4"]:
    """Compute axis-aligned bounding box per curve in pixel space.

    Args:
        control_points: (N, CP, 2) control points in [-1, 1] model space.
        H, W: Image dimensions for pixel conversion.

    Returns:
        (N, 4) tensor of [x_min, y_min, x_max, y_max] in pixel coords.
    """
    if control_points.shape[0] == 0:
        return torch.empty(0, 4, device=control_points.device, dtype=control_points.dtype)

    cp_px = model_to_pixel(control_points, H, W)  # (N, CP, 2)
    x_min = cp_px[..., 0].min(dim=-1).values  # (N,)
    y_min = cp_px[..., 1].min(dim=-1).values
    x_max = cp_px[..., 0].max(dim=-1).values
    y_max = cp_px[..., 1].max(dim=-1).values
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def compute_outside_ratio(
    aabb: Float[Tensor, "N 4"], H: int, W: int,
) -> Float[Tensor, " N"]:
    """Fraction of each curve's AABB that lies outside the image bounds.

    Args:
        aabb: (N, 4) bounding boxes [x_min, y_min, x_max, y_max] in pixels.
        H, W: Image dimensions.

    Returns:
        (N,) tensor of ratios in [0, 1]. 0 = fully inside, 1 = fully outside.
    """
    if aabb.shape[0] == 0:
        return torch.empty(0, device=aabb.device, dtype=aabb.dtype)

    x_min, y_min, x_max, y_max = aabb[:, 0], aabb[:, 1], aabb[:, 2], aabb[:, 3]

    total_width = (x_max - x_min).clamp(min=1e-8)
    total_height = (y_max - y_min).clamp(min=1e-8)
    total_area = total_width * total_height

    # Clip to image bounds
    cx_min = x_min.clamp(min=0, max=float(W))
    cy_min = y_min.clamp(min=0, max=float(H))
    cx_max = x_max.clamp(min=0, max=float(W))
    cy_max = y_max.clamp(min=0, max=float(H))

    clipped_width = (cx_max - cx_min).clamp(min=0)
    clipped_height = (cy_max - cy_min).clamp(min=0)
    clipped_area = clipped_width * clipped_height

    inside_ratio = clipped_area / total_area
    return 1.0 - inside_ratio


def compute_pairwise_iou(aabb: Float[Tensor, "N 4"]) -> Float[Tensor, "N N"]:
    """Pairwise IoU between all curve AABBs.

    Args:
        aabb: (N, 4) bounding boxes [x_min, y_min, x_max, y_max].

    Returns:
        (N, N) symmetric IoU matrix with zeros on the diagonal.
    """
    N = aabb.shape[0]
    if N == 0:
        return torch.empty(0, 0, device=aabb.device, dtype=aabb.dtype)

    x_min = aabb[:, 0]  # (N,)
    y_min = aabb[:, 1]
    x_max = aabb[:, 2]
    y_max = aabb[:, 3]

    # Intersection
    ix_min = torch.max(x_min.unsqueeze(1), x_min.unsqueeze(0))  # (N, N)
    iy_min = torch.max(y_min.unsqueeze(1), y_min.unsqueeze(0))
    ix_max = torch.min(x_max.unsqueeze(1), x_max.unsqueeze(0))
    iy_max = torch.min(y_max.unsqueeze(1), y_max.unsqueeze(0))

    iw = (ix_max - ix_min).clamp(min=0)
    ih = (iy_max - iy_min).clamp(min=0)
    intersection = iw * ih

    # Areas
    areas = (x_max - x_min) * (y_max - y_min)  # (N,)
    union = areas.unsqueeze(1) + areas.unsqueeze(0) - intersection  # (N, N)

    iou = intersection / union.clamp(min=1e-8)
    # Zero out diagonal
    iou.fill_diagonal_(0.0)
    return iou


def compute_color_distance(colors: Float[Tensor, "N 3"]) -> Float[Tensor, "N N"]:
    """Pairwise L2 distance on already-activated RGB values.

    Args:
        colors: (N, 3) colors in a comparable space (typically [0, 1]).

    Returns:
        (N, N) distance matrix.
    """
    N = colors.shape[0]
    if N == 0:
        return torch.empty(0, 0, device=colors.device, dtype=colors.dtype)

    diff = colors.unsqueeze(1) - colors.unsqueeze(0)  # (N, N, 3)
    return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  # (N, N)


def _staged_threshold(progress: float, early: float, mid: float, late: float) -> float:
    """Select early/mid/late threshold based on training progress."""
    if progress < 0.6:
        return early
    if progress < 0.8:
        return mid
    return late


# ── Composite masks ──────────────────────────────────────────────────────


def compute_overlap_suppression_mask(
    aabb: Float[Tensor, "N 4"],
    colors: Float[Tensor, "N 3"],
    areas: Float[Tensor, " N"],
    iou_threshold: float,
    color_threshold: float,
    weighted_iou_threshold: float,
    area_threshold: float,
) -> Bool[Tensor, " N"]:
    """Suppress smaller overlapping curves with similar colors.

    For each pair of overlapping + similar-color curves, the smaller one
    is a candidate for pruning. Weighted IoU accumulates across all partners.

    Args:
        aabb: (N, 4) bounding boxes.
        colors: (N, 3) colors in a comparable space (typically [0, 1]).
        areas: (N,) per-curve areas in pixel space.
        iou_threshold: Minimum IoU to consider a pair overlapping.
        color_threshold: Maximum color distance to consider similar.
        weighted_iou_threshold: Accumulated weighted IoU threshold for pruning.
        area_threshold: Only prune curves smaller than this area.

    Returns:
        (N,) boolean keep mask (True = keep, False = prune).
    """
    N = aabb.shape[0]
    if N == 0:
        return torch.ones(0, device=aabb.device, dtype=torch.bool)

    iou = compute_pairwise_iou(aabb)  # (N, N)
    cdist = compute_color_distance(colors)  # (N, N)

    # Pairs that overlap AND have similar color
    overlap_similar = (iou > iou_threshold) & (cdist < color_threshold)

    # For each curve, compute weighted IoU sum — weighted by how much the
    # partner's area dominates. Larger partners suppress smaller curves.
    # weight_ij = area_j / (area_i + area_j + eps)
    area_i = areas.unsqueeze(1).expand(N, N)
    area_j = areas.unsqueeze(0).expand(N, N)
    weight = area_j / (area_i + area_j + 1e-8)

    weighted_iou = (iou * weight * overlap_similar.float()).sum(dim=-1)  # (N,)

    # Mark for pruning: weighted IoU exceeds threshold AND curve is small
    prune = (weighted_iou > weighted_iou_threshold) & (areas < area_threshold)
    return ~prune


def compute_tiny_curve_mask(
    aabb: Float[Tensor, "N 4"],
    width_thresh: float,
    height_thresh: float,
    area_thresh: float,
) -> Bool[Tensor, " N"]:
    """Mark curves whose AABB is smaller than all three thresholds.

    A curve is pruned only if its width AND height AND area are ALL below
    the respective thresholds.

    Args:
        aabb: (N, 4) bounding boxes [x_min, y_min, x_max, y_max].
        width_thresh: Minimum AABB width in pixels.
        height_thresh: Minimum AABB height in pixels.
        area_thresh: Minimum AABB area in pixels^2.

    Returns:
        (N,) boolean keep mask (True = keep, False = prune).
    """
    if aabb.shape[0] == 0:
        return torch.ones(0, device=aabb.device, dtype=torch.bool)

    w = aabb[:, 2] - aabb[:, 0]
    h = aabb[:, 3] - aabb[:, 1]
    a = w * h

    is_tiny = (w < width_thresh) & (h < height_thresh) & (a < area_thresh)
    return ~is_tiny


# ── Main pruning entry points ───────────────────────────────────────────


def compute_prune_mask_open(
    scene: VectorGraphicsScene,
    progress: float,
    config: PruneConfig,
    H: int,
    W: int,
) -> tuple[Tensor, list[dict]]:
    """Compute keep mask for open curves.

    Combines: outside_ratio, overlap_suppression, tiny_curve, opacity threshold.

    Args:
        scene: The current scene.
        progress: Training progress in [0, 1] (step / total_steps).
        config: Pruning configuration.
        H, W: Image dimensions.

    Returns:
        keep_mask: (N,) boolean tensor — True = keep the curve.
        metrics: List of per-curve dicts with diagnostic info.
    """
    device = scene.open_control_points.device
    N = scene.n_open
    if N == 0:
        return torch.ones(0, device=device, dtype=torch.bool), []

    cp = scene.open_control_points  # (N, 10, 2)

    # Compute AABB from control points
    aabb = compute_aabb(cp, H, W)  # (N, 4)

    # 1. Outside ratio
    outside = compute_outside_ratio(aabb, H, W)  # (N,)
    keep_outside = outside <= config.outside_ratio_threshold

    # 2. Opacity: keep if any segment has opacity above threshold
    open_opacities = torch.sigmoid(scene.open_opacities)  # (N, 3)
    max_opacity = open_opacities.max(dim=-1).values  # (N,)
    # Decaying threshold: starts strict at the configured value, decays linearly
    opacity_thresh = config.opacity_threshold_open * (1 - progress) + 0.01 * progress
    keep_opacity = max_opacity > opacity_thresh

    # 3. Area proxy (arc length x stroke width)
    cp_px = model_to_pixel(cp, H, W)
    edge_lengths = torch.sqrt(
        ((cp_px[:, 1:] - cp_px[:, :-1]) ** 2).sum(dim=-1) + 1e-12,
    ).sum(dim=-1)
    stroke_w = 0.5 + torch.sigmoid(scene.open_stroke_widths) * 4.5
    areas = edge_lengths * stroke_w  # (N,)

    # 4. Tiny curve suppression
    keep_tiny = compute_tiny_curve_mask(
        aabb,
        config.tiny_width_open,
        config.tiny_height_open,
        config.tiny_area_open,
    )

    # 5. Overlap + color similarity suppression
    # Select area threshold based on progress
    overlap_area_thresh = _staged_threshold(
        progress,
        config.area_threshold_early,
        config.area_threshold_mid,
        config.area_threshold_late,
    )

    open_overlap_colors = scene.open_colors.clamp(0.0, 1.0) * open_opacities
    keep_overlap = compute_overlap_suppression_mask(
        aabb,
        open_overlap_colors,
        areas,
        config.iou_threshold_open,
        config.color_threshold_open,
        config.weighted_iou_threshold,
        overlap_area_thresh,
    )

    # Combined mask
    keep_mask = keep_outside & keep_opacity & keep_tiny & keep_overlap

    metrics = _build_prune_metrics(
        aabb, outside, max_opacity, areas,
        keep_mask, keep_outside, keep_opacity, keep_tiny, keep_overlap,
    )

    return keep_mask, metrics


def _build_prune_metrics(
    aabb: Tensor,
    outside: Tensor,
    opacity: Tensor,
    areas: Tensor,
    keep_mask: Tensor,
    keep_outside: Tensor,
    keep_opacity: Tensor,
    keep_tiny: Tensor,
    keep_overlap: Tensor,
) -> list[dict]:
    """Assemble per-curve diagnostic dicts with a single host sync per tensor."""
    max_iou = compute_pairwise_iou(aabb).max(dim=-1).values if aabb.shape[0] > 0 else aabb.new_zeros(0)

    outside_l = outside.tolist()
    max_iou_l = max_iou.tolist()
    opacity_l = opacity.tolist()
    areas_l = areas.tolist()
    keep_l = keep_mask.tolist()
    ko_l = keep_outside.tolist()
    kop_l = keep_opacity.tolist()
    kt_l = keep_tiny.tolist()
    kov_l = keep_overlap.tolist()

    metrics: list[dict] = []
    for i in range(len(keep_l)):
        pruned = not keep_l[i]
        reason = ""
        if pruned:
            if not ko_l[i]:
                reason = "outside_image"
            elif not kop_l[i]:
                reason = "low_opacity"
            elif not kt_l[i]:
                reason = "tiny_curve"
            elif not kov_l[i]:
                reason = "overlap_suppressed"
        metrics.append({
            "outside_ratio": outside_l[i],
            "max_iou": max_iou_l[i],
            "opacity": opacity_l[i],
            "area": areas_l[i],
            "pruned": pruned,
            "reason": reason,
        })
    return metrics


def compute_prune_mask_closed(
    scene: VectorGraphicsScene,
    progress: float,
    config: PruneConfig,
    H: int,
    W: int,
) -> tuple[Tensor, list[dict]]:
    """Compute keep mask for closed curves.

    Same signals as open curves but with closed-curve-specific thresholds
    and staged area thresholds based on progress.

    Args:
        scene: The current scene.
        progress: Training progress in [0, 1] (step / total_steps).
        config: Pruning configuration.
        H, W: Image dimensions.

    Returns:
        keep_mask: (N,) boolean tensor — True = keep the curve.
        metrics: List of per-curve dicts with diagnostic info.
    """
    device = scene.closed_shared_pts.device
    N = scene.n_closed
    if N == 0:
        return torch.ones(0, device=device, dtype=torch.bool), []

    bcp = scene.closed_boundary_cp  # (N, 2, num_cp, 2)
    # Flatten boundaries for AABB: take min/max across both boundaries
    # Shape (N, 2*num_cp, 2)
    num_cp = bcp.shape[2]
    cp_flat = bcp.reshape(N, 2 * num_cp, 2)
    aabb = compute_aabb(cp_flat, H, W)  # (N, 4)

    # 1. Outside ratio
    outside = compute_outside_ratio(aabb, H, W)
    keep_outside = outside <= config.outside_ratio_threshold

    # 2. Opacity (staged threshold).
    # Legacy checkpoints may provide scalar opacity; current profile uses (N, 3).
    closed_op = torch.sigmoid(scene.closed_opacities)
    if closed_op.ndim == 2:
        closed_opacity_strength = closed_op.sum(dim=-1)  # official-style criterion
    else:
        closed_opacity_strength = closed_op
    if progress < 0.7:
        opacity_thresh = config.opacity_threshold_closed_early
    else:
        opacity_thresh = config.opacity_threshold_closed_late
    keep_opacity = closed_opacity_strength > opacity_thresh

    # 3. True enclosed area
    bcp_px = model_to_pixel(bcp, H, W)
    areas = closed_curve_enclosed_area(bcp_px)  # (N,)

    # Staged area threshold
    area_thresh = _staged_threshold(
        progress,
        config.area_threshold_early,
        config.area_threshold_mid,
        config.area_threshold_late,
    )

    # 4. Tiny curve suppression
    keep_tiny = compute_tiny_curve_mask(
        aabb,
        config.tiny_width_closed,
        config.tiny_height_closed,
        config.tiny_area_closed,
    )

    # 5. Overlap + color similarity suppression
    closed_opacity_rgb = closed_op
    if closed_opacity_rgb.ndim == 1:
        closed_opacity_rgb = closed_opacity_rgb[:, None]
    if closed_opacity_rgb.shape[1] == 1:
        closed_opacity_rgb = closed_opacity_rgb.expand(-1, 3)
    elif closed_opacity_rgb.shape[1] > 3:
        closed_opacity_rgb = closed_opacity_rgb[:, :3]
    closed_overlap_colors = torch.sigmoid(scene.closed_colors) * closed_opacity_rgb
    keep_overlap = compute_overlap_suppression_mask(
        aabb,
        closed_overlap_colors,
        areas,
        config.iou_threshold_closed,
        config.color_threshold_closed,
        config.weighted_iou_threshold,
        area_thresh,
    )

    # Combined mask
    keep_mask = keep_outside & keep_opacity & keep_tiny & keep_overlap

    metrics = _build_prune_metrics(
        aabb, outside, closed_opacity_strength, areas,
        keep_mask, keep_outside, keep_opacity, keep_tiny, keep_overlap,
    )

    return keep_mask, metrics


# ── Densification ────────────────────────────────────────────────────────


def compute_densify_targets(
    rendered: Float[Tensor, "3 H W"],
    target: Float[Tensor, "3 H W"],
    n_new: int,
    H: int,
    W: int,
    nodiff_threshold: float = 0.05,
) -> tuple[Float[Tensor, "M 2"], Float[Tensor, " M"]]:
    """Find error-hotspot centers and shapes for curve densification.

    Computes per-pixel squared error, zeros out low-error regions, then finds
    the top ``n_new`` hotspot cells via grid partitioning. Cells with zero
    remaining error are dropped. For each selected cell, an aspect ratio of
    its high-error sub-region is computed so the caller can pick a curve type
    (elongated regions suit open strokes, blob-like regions suit closed fills).

    Args:
        rendered: (3, H, W) rendered image.
        target: (3, H, W) target image.
        n_new: Number of new curve centers to find.
        H, W: Image dimensions.
        nodiff_threshold: Zero errors below this squared-error threshold.

    Returns:
        Tuple of:
            - (M, 2) pixel-space centers as (x, y), where M <= n_new.
            - (M,) aspect ratios (>= 1) of the high-error sub-region per cell.
    """
    device = rendered.device
    if n_new <= 0:
        return torch.empty(0, 2, device=device), torch.empty(0, device=device)

    # Per-pixel error: sum across channels, zeroing low-error regions
    error_map = ((rendered - target) ** 2).sum(dim=0)  # (H, W)
    error_map = error_map * (error_map > nodiff_threshold).float()

    # Grid-cell partitioning (vectorized cell means via zero padding)
    cell_size = max(1, max(H, W) // 8)
    n_cells_y = (H + cell_size - 1) // cell_size
    n_cells_x = (W + cell_size - 1) // cell_size

    padded = torch.nn.functional.pad(
        error_map, (0, n_cells_x * cell_size - W, 0, n_cells_y * cell_size - H),
    )
    cell_sums = padded.view(n_cells_y, cell_size, n_cells_x, cell_size).sum(dim=(1, 3))
    heights = torch.full((n_cells_y,), cell_size, device=device, dtype=error_map.dtype)
    heights[-1] = H - cell_size * (n_cells_y - 1)
    widths = torch.full((n_cells_x,), cell_size, device=device, dtype=error_map.dtype)
    widths[-1] = W - cell_size * (n_cells_x - 1)
    cell_means = cell_sums / (heights[:, None] * widths[None, :])

    flat_means = cell_means.reshape(-1)
    k = min(n_new, flat_means.numel())
    top_vals, top_idx = torch.topk(flat_means, k)
    top_idx = top_idx[top_vals > 0]  # drop cells with no residual error

    centers: list[tuple[float, float]] = []
    aspects: list[float] = []
    for idx in top_idx.tolist():
        cy, cx = divmod(idx, n_cells_x)
        y0, x0 = cy * cell_size, cx * cell_size
        y1, x1 = min(y0 + cell_size, H), min(x0 + cell_size, W)
        centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))

        # Aspect ratio of the high-error sub-region within the cell
        err_patch = error_map[y0:y1, x0:x1]
        patch_max = err_patch.max()
        high_err = err_patch > patch_max * 0.5
        ys, xs = torch.where(high_err)
        if ys.numel() > 0:
            span_y = (ys.max() - ys.min() + 1).item()
            span_x = (xs.max() - xs.min() + 1).item()
            aspects.append(max(span_x, span_y) / max(1, min(span_x, span_y)))
        else:
            aspects.append(1.0)

    centers_t = torch.tensor(centers, device=device, dtype=torch.float32).reshape(-1, 2)
    aspects_t = torch.tensor(aspects, device=device, dtype=torch.float32)
    return centers_t, aspects_t


def compute_densify_centers(
    rendered: Float[Tensor, "3 H W"],
    target: Float[Tensor, "3 H W"],
    n_new: int,
    H: int,
    W: int,
    nodiff_threshold: float = 0.05,
) -> Float[Tensor, "M 2"]:
    """Find error-hotspot centers for curve densification.

    Thin wrapper around :func:`compute_densify_targets` that returns only the
    centers.

    Returns:
        (M, 2) pixel-space centers as (x, y), where M <= n_new.
    """
    centers, _ = compute_densify_targets(rendered, target, n_new, H, W, nodiff_threshold)
    return centers
