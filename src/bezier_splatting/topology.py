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
    """Pairwise L2 distance on sigmoid(colors).

    Args:
        colors: (N, 3) pre-sigmoid color values.

    Returns:
        (N, N) distance matrix.
    """
    N = colors.shape[0]
    if N == 0:
        return torch.empty(0, 0, device=colors.device, dtype=colors.dtype)

    c = torch.sigmoid(colors)  # (N, 3)
    diff = c.unsqueeze(1) - c.unsqueeze(0)  # (N, N, 3)
    return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  # (N, N)


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
        colors: (N, 3) pre-sigmoid colors.
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
    if progress < 0.6:
        overlap_area_thresh = config.area_threshold_early
    elif progress < 0.8:
        overlap_area_thresh = config.area_threshold_mid
    else:
        overlap_area_thresh = config.area_threshold_late

    keep_overlap = compute_overlap_suppression_mask(
        aabb,
        scene.open_colors,
        areas,
        config.iou_threshold_open,
        config.color_threshold_open,
        config.weighted_iou_threshold,
        overlap_area_thresh,
    )

    # Combined mask
    keep_mask = keep_outside & keep_opacity & keep_tiny & keep_overlap

    # Build per-curve metrics
    iou_matrix = compute_pairwise_iou(aabb)
    max_iou = iou_matrix.max(dim=-1).values if N > 0 else torch.zeros(0, device=device)

    metrics: list[dict] = []
    for i in range(N):
        pruned = not keep_mask[i].item()
        reason = ""
        if pruned:
            if not keep_outside[i].item():
                reason = "outside_image"
            elif not keep_opacity[i].item():
                reason = "low_opacity"
            elif not keep_tiny[i].item():
                reason = "tiny_curve"
            elif not keep_overlap[i].item():
                reason = "overlap_suppressed"
        metrics.append({
            "outside_ratio": outside[i].item(),
            "max_iou": max_iou[i].item(),
            "opacity": max_opacity[i].item(),
            "area": areas[i].item(),
            "pruned": pruned,
            "reason": reason,
        })

    return keep_mask, metrics


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

    # 2. Opacity (staged threshold)
    closed_opacities = torch.sigmoid(scene.closed_opacities)  # (N,)
    if progress < 0.7:
        opacity_thresh = config.opacity_threshold_closed_early
    else:
        opacity_thresh = config.opacity_threshold_closed_late
    keep_opacity = closed_opacities > opacity_thresh

    # 3. True enclosed area
    bcp_px = model_to_pixel(bcp, H, W)
    areas = closed_curve_enclosed_area(bcp_px)  # (N,)

    # Staged area threshold
    if progress < 0.6:
        area_thresh = config.area_threshold_early
    elif progress < 0.8:
        area_thresh = config.area_threshold_mid
    else:
        area_thresh = config.area_threshold_late

    # 4. Tiny curve suppression
    keep_tiny = compute_tiny_curve_mask(
        aabb,
        config.tiny_width_closed,
        config.tiny_height_closed,
        config.tiny_area_closed,
    )

    # 5. Overlap + color similarity suppression
    keep_overlap = compute_overlap_suppression_mask(
        aabb,
        scene.closed_colors,
        areas,
        config.iou_threshold_closed,
        config.color_threshold_closed,
        config.weighted_iou_threshold,
        area_thresh,
    )

    # Combined mask
    keep_mask = keep_outside & keep_opacity & keep_tiny & keep_overlap

    # Build per-curve metrics
    iou_matrix = compute_pairwise_iou(aabb)
    max_iou = iou_matrix.max(dim=-1).values if N > 0 else torch.zeros(0, device=device)

    metrics: list[dict] = []
    for i in range(N):
        pruned = not keep_mask[i].item()
        reason = ""
        if pruned:
            if not keep_outside[i].item():
                reason = "outside_image"
            elif not keep_opacity[i].item():
                reason = "low_opacity"
            elif not keep_tiny[i].item():
                reason = "tiny_curve"
            elif not keep_overlap[i].item():
                reason = "overlap_suppressed"
        metrics.append({
            "outside_ratio": outside[i].item(),
            "max_iou": max_iou[i].item(),
            "opacity": closed_opacities[i].item(),
            "area": areas[i].item(),
            "pruned": pruned,
            "reason": reason,
        })

    return keep_mask, metrics


# ── Densification ────────────────────────────────────────────────────────


def compute_densify_centers(
    rendered: Float[Tensor, "3 H W"],
    target: Float[Tensor, "3 H W"],
    n_new: int,
    H: int,
    W: int,
    nodiff_threshold: float = 0.05,
) -> Float[Tensor, "M 2"]:
    """Find error-hotspot centers for curve densification.

    Computes per-pixel squared error, zeros out low-error regions, then
    finds the top ``n_new`` hotspot centers via grid-cell partitioning.

    Args:
        rendered: (3, H, W) rendered image.
        target: (3, H, W) target image.
        n_new: Number of new curve centers to find.
        H, W: Image dimensions.
        nodiff_threshold: Zero errors below this squared-error threshold.

    Returns:
        (M, 2) pixel-space centers as (x, y), where M <= n_new.
    """
    device = rendered.device
    if n_new <= 0:
        return torch.empty(0, 2, device=device)

    # Per-pixel error: sum across channels
    error_map = ((rendered - target) ** 2).sum(dim=0)  # (H, W)

    # Zero out low-error regions
    error_map = error_map * (error_map > nodiff_threshold).float()

    # Grid-cell partitioning to find hotspot centers
    cell_size = max(H, W) // 8
    n_cells_y = (H + cell_size - 1) // cell_size
    n_cells_x = (W + cell_size - 1) // cell_size

    cell_errors: list[float] = []
    cell_centers: list[tuple[float, float]] = []

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            x0 = cx * cell_size
            y1 = min(y0 + cell_size, H)
            x1 = min(x0 + cell_size, W)
            err_patch = error_map[y0:y1, x0:x1]
            cell_errors.append(err_patch.mean().item())
            cell_centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))

    sorted_indices = sorted(
        range(len(cell_errors)), key=lambda i: cell_errors[i], reverse=True,
    )
    top_indices = sorted_indices[:n_new]

    centers = torch.tensor(
        [cell_centers[i] for i in top_indices],
        device=device,
        dtype=torch.float32,
    )
    return centers  # (M, 2) — pixel-space (x, y)
