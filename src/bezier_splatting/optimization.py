"""Training loop with adaptive pruning and densification.

Implements the LIVE Xing loss for self-intersection prevention and
paper-aligned pruning/densification with StepLR decay.

All control points are stored in [0, 1] normalized coordinates.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .area import closed_curve_enclosed_area
from .model import VectorGraphicsScene


# ── Xing Loss (LIVE method) ─────────────────────────────────────────────


def _sine_theta(a: Tensor, b: Tensor) -> Tensor:
    """Signed sine of the angle between 2D vector pairs.

    Args:
        a, b: (N, 2) vectors.

    Returns:
        sin(θ) for each pair, shape (N,).
    """
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    norms = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
    return cross / norms


def _xing_loss_cubic(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """LIVE Xing loss for a batch of cubic Bézier segments.

    Penalizes self-intersecting control polygons using direction-gated
    sine penalty: if the middle CP crosses to the wrong side of the
    chord (p0→p3), penalize.

    Args:
        p0, p1, p2, p3: Control points, each (N, 2).

    Returns:
        Per-segment loss (N,).
    """
    cs1 = p1 - p0  # (N, 2)
    cs2 = p2 - p0
    cs3 = p3 - p0

    sina = _sine_theta(cs1, cs3)    # sine of angle between first edge and chord
    sin12 = _sine_theta(cs1, cs2)   # sine of angle between first edge and second edge

    direct = (sin12 >= 0).float()
    opst = 1.0 - direct

    loss = direct * F.relu(-sina) + opst * F.relu(sina)
    return loss


def _xing_loss(scene: VectorGraphicsScene) -> Tensor:
    """Total Xing loss for all curves in the scene.

    Open curves: 3 cubic segments each (CPs [0:4], [3:7], [6:10]).
    Closed curves: 1 cubic per boundary × 2 boundaries (when 4 CPs),
        or sliding window of cubics for higher-order boundaries.
    """
    losses: list[Tensor] = []

    # Open curves: 3 segments per curve
    if scene.n_open > 0:
        cp = scene.open_control_points  # (N, 10, 2)
        for seg_start in [0, 3, 6]:
            loss = _xing_loss_cubic(
                cp[:, seg_start], cp[:, seg_start + 1],
                cp[:, seg_start + 2], cp[:, seg_start + 3],
            )
            losses.append(loss)

    # Closed curves: per-boundary cubics
    if scene.n_closed > 0:
        bcp = scene.closed_boundary_cp  # (N, 2, num_cp, 2)
        num_cp = bcp.shape[2]
        if num_cp == 4:
            for b in range(2):
                loss = _xing_loss_cubic(
                    bcp[:, b, 0], bcp[:, b, 1],
                    bcp[:, b, 2], bcp[:, b, 3],
                )
                losses.append(loss)
        elif num_cp > 4:
            # Sliding window of cubics along each boundary
            for b in range(2):
                for i in range(num_cp - 3):
                    loss = _xing_loss_cubic(
                        bcp[:, b, i], bcp[:, b, i + 1],
                        bcp[:, b, i + 2], bcp[:, b, i + 3],
                    )
                    losses.append(loss)

    if not losses:
        device = next(scene.parameters()).device
        return torch.tensor(0.0, device=device)

    return torch.cat(losses).sum()


# ── Training Loop ────────────────────────────────────────────────────────


def fit_image(
    target: Tensor,
    n_open: int = 128,
    n_closed: int = 64,
    steps: int = 15000,
    prune_every: int = 400,
    prune_stop_before_end: int = 1000,
    lambda_xing: float = 0.01,
    log_every: int = 100,
    lr_step_size: int = 5000,
    lr_gamma: float = 0.5,
    callback: Callable[[int, float, VectorGraphicsScene], None] | None = None,
) -> VectorGraphicsScene:
    """Optimize a VectorGraphicsScene to reconstruct a target image.

    Args:
        target: (3, H, W) target image in [0, 1].
        n_open: Initial number of open curves.
        n_closed: Initial number of closed curves.
        steps: Total optimization steps.
        prune_every: Prune/densify every N steps.
        prune_stop_before_end: Stop pruning this many steps before the end.
        lambda_xing: Weight for LIVE Xing loss.
        log_every: Print loss every N steps.
        lr_step_size: StepLR decay period.
        lr_gamma: StepLR multiplicative decay factor.
        callback: Optional callback(step, loss, scene) for monitoring.

    Returns:
        Optimized VectorGraphicsScene.
    """
    _, H, W = target.shape
    device = target.device

    scene = VectorGraphicsScene(
        n_open=n_open, n_closed=n_closed, H=H, W=W,
    ).to(device)

    param_groups = _build_param_groups(scene, H, W)
    optimizer = torch.optim.Adam(param_groups)

    loss_history: list[float] = []
    lr_decay = 1.0  # accumulated decay factor

    for step in range(steps):
        # StepLR-style decay
        if step > 0 and step % lr_step_size == 0:
            lr_decay *= lr_gamma
            for group in optimizer.param_groups:
                group["lr"] *= lr_gamma

        optimizer.zero_grad()

        rendered = scene(H, W)
        mse_loss = F.mse_loss(rendered, target)

        xing_loss = _xing_loss(scene)

        loss = mse_loss + lambda_xing * xing_loss
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % log_every == 0:
            print(f"Step {step:5d}/{steps} | loss={loss_val:.6f} | MSE={mse_loss.item():.6f}")

        if callback is not None:
            callback(step, loss_val, scene)

        # Pruning and densification
        should_prune = (
            step > 0
            and step % prune_every == 0
            and step < steps - prune_stop_before_end
        )
        if should_prune:
            with torch.no_grad():
                _prune_and_densify(scene, target, rendered, step, steps, H, W)
            # Rebuild optimizer (fresh Adam state for new params)
            param_groups = _build_param_groups(scene, H, W)
            optimizer = torch.optim.Adam(param_groups)
            # Apply accumulated lr decay to new optimizer
            for group in optimizer.param_groups:
                group["lr"] *= lr_decay

    return scene


def _build_param_groups(scene: VectorGraphicsScene, H: int, W: int) -> list[dict]:
    """Build optimizer parameter groups with per-type learning rates.

    Control points are in [0, 1] normalized coordinates. The CP learning rate
    is scaled by resolution so the effective pixel displacement is ~0.25 px/iter
    regardless of image size.
    """
    cp_lr = 0.25 / max(H, W)
    groups: list[dict] = []

    if scene.n_open > 0:
        groups.extend([
            {"params": [scene.open_control_points], "lr": cp_lr, "name": "open_cp"},
            {"params": [scene.open_colors], "lr": 0.01, "name": "open_colors"},
            {"params": [scene.open_opacities], "lr": 0.1, "name": "open_opacities"},
            {"params": [scene.open_stroke_widths], "lr": 0.05, "name": "open_stroke_widths"},
        ])

    if scene.n_closed > 0:
        groups.extend([
            {"params": [scene.closed_boundary_cp], "lr": cp_lr, "name": "closed_cp"},
            {"params": [scene.closed_colors], "lr": 0.01, "name": "closed_colors"},
            {"params": [scene.closed_opacities], "lr": 0.1, "name": "closed_opacities"},
        ])

    return groups


# ── Pruning & Densification ─────────────────────────────────────────────


def _prune_and_densify(
    scene: VectorGraphicsScene,
    target: Tensor,
    rendered: Tensor,
    step: int,
    total_steps: int,
    H: int,
    W: int,
) -> None:
    """Prune low-opacity/small-area curves and densify at high-error regions.

    Modifies scene in-place via nn.Parameter assignment.

    Strategy (paper Algorithm 1):
        1. Prune curves with opacity below threshold or negligible area
        2. Find high-error regions in the reconstruction
        3. Insert new curves at error hotspots, matching type and color from target
    """
    device = target.device

    # Decaying opacity threshold: starts strict, becomes lenient
    progress = step / total_steps
    opacity_threshold = 0.3 * (1 - progress) + 0.01 * progress

    # ── Prune open curves ──
    pruned_open = 0
    if scene.n_open > 0:
        # Per-segment opacity: keep curve if any segment is visible
        open_opacities = torch.sigmoid(scene.open_opacities)  # (N, 3)
        max_opacity = open_opacities.max(dim=-1).values  # (N,)

        # Area proxy: arc length × stroke width in pixel space
        cp = scene.open_control_points  # (N, 10, 2) in [0,1]
        cp_px = cp * torch.tensor([W, H], device=device, dtype=cp.dtype)
        edge_lengths = torch.norm(cp_px[:, 1:] - cp_px[:, :-1], dim=-1).sum(dim=-1)
        stroke_w = 0.5 + torch.sigmoid(scene.open_stroke_widths) * 4.5
        areas = edge_lengths * stroke_w

        keep_mask = (max_opacity > opacity_threshold) & (areas > 1.0)
        pruned_open = (~keep_mask).sum().item()

        if pruned_open > 0:
            if keep_mask.any():
                scene.open_control_points = nn.Parameter(scene.open_control_points[keep_mask].clone())
                scene.open_colors = nn.Parameter(scene.open_colors[keep_mask].clone())
                scene.open_opacities = nn.Parameter(scene.open_opacities[keep_mask].clone())
                scene.open_stroke_widths = nn.Parameter(scene.open_stroke_widths[keep_mask].clone())
                scene.n_open = keep_mask.sum().item()
            else:
                scene.open_control_points = nn.Parameter(torch.empty(0, 10, 2, device=device))
                scene.open_colors = nn.Parameter(torch.empty(0, 3, device=device))
                scene.open_opacities = nn.Parameter(torch.empty(0, 3, device=device))
                scene.open_stroke_widths = nn.Parameter(torch.empty(0, device=device))
                scene.n_open = 0

    # ── Prune closed curves ──
    pruned_closed = 0
    if scene.n_closed > 0:
        closed_opacities = torch.sigmoid(scene.closed_opacities)  # (N,)

        # True enclosed area in pixel space
        bcp = scene.closed_boundary_cp  # (N, 2, num_cp, 2) in [0,1]
        bcp_px = bcp * torch.tensor([W, H], device=device, dtype=bcp.dtype)
        areas = closed_curve_enclosed_area(bcp_px)

        keep_mask = (closed_opacities > opacity_threshold) & (areas > 4.0)
        pruned_closed = (~keep_mask).sum().item()

        if pruned_closed > 0:
            if keep_mask.any():
                scene.closed_boundary_cp = nn.Parameter(scene.closed_boundary_cp[keep_mask].clone())
                scene.closed_colors = nn.Parameter(scene.closed_colors[keep_mask].clone())
                scene.closed_opacities = nn.Parameter(scene.closed_opacities[keep_mask].clone())
                scene.n_closed = keep_mask.sum().item()
            else:
                num_cp = scene.closed_boundary_cp.shape[2]
                scene.closed_boundary_cp = nn.Parameter(torch.empty(0, 2, num_cp, 2, device=device))
                scene.closed_colors = nn.Parameter(torch.empty(0, 3, device=device))
                scene.closed_opacities = nn.Parameter(torch.empty(0, device=device))
                scene.n_closed = 0

    total_pruned = pruned_open + pruned_closed

    # ── Densification ──
    if total_pruned > 0:
        error_map = (rendered - target).abs().mean(dim=0)  # (H, W)
        _densify_curves(scene, error_map, target, total_pruned, H, W, device)


def _densify_curves(
    scene: VectorGraphicsScene,
    error_map: Tensor,
    target: Tensor,
    n_new: int,
    H: int,
    W: int,
    device: torch.device,
) -> None:
    """Insert new curves at high-error regions.

    Inserts a mix of open and closed curves based on error region shape.
    Colors are initialized from the target image (logit-space for the model).

    Args:
        scene: Scene to modify in-place.
        error_map: Per-pixel error (H, W).
        target: Target image (3, H, W) for color initialization.
        n_new: Number of new curves to insert.
        H, W: Image dimensions.
        device: Torch device.
    """
    if n_new <= 0:
        return

    # Find error hotspots via grid cells
    cell_size = max(H, W) // 8
    n_cells_y = (H + cell_size - 1) // cell_size
    n_cells_x = (W + cell_size - 1) // cell_size

    cell_errors: list[float] = []
    cell_centers: list[tuple[float, float]] = []
    cell_aspect: list[float] = []

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            x0 = cx * cell_size
            y1 = min(y0 + cell_size, H)
            x1 = min(x0 + cell_size, W)
            err_patch = error_map[y0:y1, x0:x1]
            cell_errors.append(err_patch.mean().item())
            cell_centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))

            # Aspect ratio of high-error sub-region → determines curve type
            if err_patch.max() > 0:
                thresh = err_patch.max() * 0.5
                high_err = err_patch > thresh
                if high_err.any():
                    ys, xs = torch.where(high_err)
                    span_y = (ys.max() - ys.min() + 1).item()
                    span_x = (xs.max() - xs.min() + 1).item()
                    cell_aspect.append(max(span_x, span_y) / (min(span_x, span_y) + 1))
                else:
                    cell_aspect.append(1.0)
            else:
                cell_aspect.append(1.0)

    sorted_indices = sorted(
        range(len(cell_errors)), key=lambda i: cell_errors[i], reverse=True,
    )
    top_indices = sorted_indices[:n_new]

    closed_cp_count = scene.closed_boundary_cp.shape[2]

    new_open_cps: list[Tensor] = []
    new_open_colors: list[Tensor] = []
    new_closed_cps: list[Tensor] = []
    new_closed_colors: list[Tensor] = []

    for idx in top_indices:
        cx_px, cy_px = cell_centers[idx]
        aspect = cell_aspect[idx]

        # Initialize color from target at center (logit space)
        px = int(min(max(cx_px, 0), W - 1))
        py = int(min(max(cy_px, 0), H - 1))
        color = target[:, py, px].detach().clone()
        color = torch.clamp(color, 0.01, 0.99)
        color = torch.log(color / (1.0 - color))  # logit (inverse sigmoid)

        # High aspect → edge-like → open curve, else closed curve
        if aspect > 2.0 or scene.n_closed == 0:
            _make_open_curve(cx_px, cy_px, H, W, color, device, new_open_cps, new_open_colors)
        else:
            _make_closed_curve(
                cx_px, cy_px, H, W, closed_cp_count, color, device,
                new_closed_cps, new_closed_colors,
            )

    # ── Append new open curves ──
    if new_open_cps:
        new_cps_t = torch.stack(new_open_cps)
        new_colors_t = torch.stack(new_open_colors)
        m = len(new_open_cps)
        new_opacities = torch.zeros(m, 3, device=device)
        new_stroke_widths = torch.zeros(m, device=device)

        if scene.n_open > 0:
            scene.open_control_points = nn.Parameter(torch.cat([scene.open_control_points, new_cps_t]))
            scene.open_colors = nn.Parameter(torch.cat([scene.open_colors, new_colors_t]))
            scene.open_opacities = nn.Parameter(torch.cat([scene.open_opacities, new_opacities]))
            scene.open_stroke_widths = nn.Parameter(torch.cat([scene.open_stroke_widths, new_stroke_widths]))
        else:
            scene.open_control_points = nn.Parameter(new_cps_t)
            scene.open_colors = nn.Parameter(new_colors_t)
            scene.open_opacities = nn.Parameter(new_opacities)
            scene.open_stroke_widths = nn.Parameter(new_stroke_widths)
        scene.n_open = scene.open_control_points.shape[0]

    # ── Append new closed curves ──
    if new_closed_cps:
        new_bcp_t = torch.stack(new_closed_cps)
        new_colors_t = torch.stack(new_closed_colors)
        m = len(new_closed_cps)
        new_opacities = torch.zeros(m, device=device)

        if scene.n_closed > 0:
            scene.closed_boundary_cp = nn.Parameter(torch.cat([scene.closed_boundary_cp, new_bcp_t]))
            scene.closed_colors = nn.Parameter(torch.cat([scene.closed_colors, new_colors_t]))
            scene.closed_opacities = nn.Parameter(torch.cat([scene.closed_opacities, new_opacities]))
        else:
            scene.closed_boundary_cp = nn.Parameter(new_bcp_t)
            scene.closed_colors = nn.Parameter(new_colors_t)
            scene.closed_opacities = nn.Parameter(new_opacities)
        scene.n_closed = scene.closed_boundary_cp.shape[0]


def _make_open_curve(
    cx_px: float, cy_px: float,
    H: int, W: int,
    color: Tensor, device: torch.device,
    out_cps: list[Tensor], out_colors: list[Tensor],
) -> None:
    """Create one new open curve in [0, 1] coords near a pixel center."""
    cx_n = cx_px / W
    cy_n = cy_px / H
    spread = 0.05

    cp = torch.zeros(10, 2, device=device)
    t_vals = torch.linspace(-1, 1, 10, device=device)
    cp[:, 0] = cx_n + t_vals * spread + torch.randn(10, device=device) * spread * 0.3
    cp[:, 1] = cy_n + torch.randn(10, device=device) * spread * 0.5
    cp = cp.clamp(0, 1)

    out_cps.append(cp)
    out_colors.append(color)


def _make_closed_curve(
    cx_px: float, cy_px: float,
    H: int, W: int,
    num_cp: int, color: Tensor, device: torch.device,
    out_cps: list[Tensor], out_colors: list[Tensor],
) -> None:
    """Create one new closed curve in [0, 1] coords near a pixel center."""
    cx_n = cx_px / W
    cy_n = cy_px / H
    size = 0.04

    bcp = torch.zeros(2, num_cp, 2, device=device)
    t = torch.linspace(0, 1, num_cp, device=device)
    for b in range(2):
        y_off = size * (1 if b == 0 else -1)
        bcp[b, :, 0] = cx_n + (t - 0.5) * size * 2
        bcp[b, :, 1] = cy_n + y_off + torch.randn(num_cp, device=device) * size * 0.3

    # Shared endpoints
    shared_start = (bcp[0, 0] + bcp[1, 0]) / 2
    shared_end = (bcp[0, -1] + bcp[1, -1]) / 2
    bcp[0, 0] = shared_start
    bcp[1, 0] = shared_start
    bcp[0, -1] = shared_end
    bcp[1, -1] = shared_end
    bcp = bcp.clamp(0, 1)

    out_cps.append(bcp)
    out_colors.append(color)
