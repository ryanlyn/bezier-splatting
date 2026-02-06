"""Training loop with adaptive pruning and densification.

Uses the configurable composite loss system from losses.py.
Paper-aligned pruning/densification with StepLR decay.

All control points are stored in [-1, 1] normalized coordinates.
"""

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .adan import Adan
from .area import closed_curve_enclosed_area
from .coords import model_to_pixel, pixel_to_model
from .losses import LossConfig, compute_loss
from .model import VectorGraphicsScene
from .topology import (
    PruneConfig,
    compute_densify_centers,
    compute_prune_mask_closed,
    compute_prune_mask_open,
)


# ── Optimizer State Surgery ──────────────────────────────────────────────


def _prune_optimizer_state(
    optimizer: torch.optim.Optimizer,
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    mask: Tensor,
) -> None:
    """Slice optimizer state tensors for a pruned parameter.

    Preserves momentum for surviving curves by slicing exp_avg, exp_avg_sq,
    etc. instead of resetting to zero.

    Args:
        optimizer: The optimizer whose state to modify.
        old_param: The parameter being replaced (key in optimizer.state).
        new_param: The replacement parameter (already sliced).
        mask: Boolean mask over dim 0, True = keep.
    """
    state = optimizer.state.get(old_param)
    if state is None:
        return

    new_state: dict = {"step": state.get("step", 0)}
    for key, val in state.items():
        if key == "step":
            continue
        if isinstance(val, Tensor) and val.shape[0] == mask.shape[0]:
            new_state[key] = val[mask].clone()
        else:
            new_state[key] = val

    del optimizer.state[old_param]
    optimizer.state[new_param] = new_state

    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p is old_param:
                group["params"][i] = new_param
                return


def _extend_optimizer_state(
    optimizer: torch.optim.Optimizer,
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    n_new: int,
) -> None:
    """Extend optimizer state tensors for densified parameters.

    Appends zero-initialized state for new curves while preserving
    existing curves' momentum.

    Args:
        optimizer: The optimizer whose state to modify.
        old_param: The parameter being replaced (key in optimizer.state).
        new_param: The replacement parameter (already concatenated).
        n_new: Number of new entries appended along dim 0.
    """
    state = optimizer.state.get(old_param)
    if state is None:
        return

    new_state: dict = {"step": state.get("step", 0)}
    for key, val in state.items():
        if key == "step":
            continue
        if isinstance(val, Tensor):
            extension = torch.zeros(
                n_new, *val.shape[1:], device=val.device, dtype=val.dtype,
            )
            new_state[key] = torch.cat([val, extension], dim=0)
        else:
            new_state[key] = val

    del optimizer.state[old_param]
    optimizer.state[new_param] = new_state

    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p is old_param:
                group["params"][i] = new_param
                return


def _splice_optimizer_state(
    optimizer: torch.optim.Optimizer,
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    insert_idx: int,
    n_new: int,
) -> None:
    """Splice zero-initialized state entries into the middle of optimizer state.

    Used for the ``_depth`` parameter where new open curves are inserted between
    existing open and existing closed curve entries.

    Args:
        optimizer: The optimizer whose state to modify.
        old_param: The parameter being replaced (key in optimizer.state).
        new_param: The replacement parameter (already spliced).
        insert_idx: Index at which new entries were inserted.
        n_new: Number of new entries inserted.
    """
    state = optimizer.state.get(old_param)
    if state is None:
        return

    new_state: dict = {"step": state.get("step", 0)}
    for key, val in state.items():
        if key == "step":
            continue
        if isinstance(val, Tensor):
            before = val[:insert_idx]
            after = val[insert_idx:]
            middle = torch.zeros(
                n_new, *val.shape[1:], device=val.device, dtype=val.dtype,
            )
            new_state[key] = torch.cat([before, middle, after], dim=0)
        else:
            new_state[key] = val

    del optimizer.state[old_param]
    optimizer.state[new_param] = new_state

    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p is old_param:
                group["params"][i] = new_param
                return


# ── Training Loop ────────────────────────────────────────────────────────


def _build_optimizer(
    param_groups: list[dict],
    optimizer_type: str,
) -> torch.optim.Optimizer:
    """Create optimizer from param groups.

    Args:
        param_groups: Per-parameter-type groups with learning rates.
        optimizer_type: ``"adam"`` (default) or ``"adan"``.

    Returns:
        Configured optimizer instance.
    """
    if optimizer_type == "adan":
        return Adan(param_groups, betas=(0.98, 0.92, 0.99))
    elif optimizer_type == "adam":
        return torch.optim.Adam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type!r} (expected 'adan' or 'adam')")


def fit_image(
    target: Float[Tensor, "3 H W"],
    n_open: int = 128,
    n_closed: int = 64,
    steps: int = 15000,
    prune_every: int = 400,
    prune_stop_before_end: int = 1000,
    lambda_xing: float = 0.01,
    log_every: int = 100,
    lr_step_size: int = 5000,
    lr_gamma: float = 0.5,
    lr_scale: float = 1.0,
    optimizer_type: str = "adam",
    prune_config: PruneConfig | None = None,
    loss_config: LossConfig | None = None,
    callback: Callable[[int, float, VectorGraphicsScene], None] | None = None,
    debug: bool | str = False,
) -> VectorGraphicsScene:
    """Optimize a VectorGraphicsScene to reconstruct a target image.

    Args:
        target: (3, H, W) target image in [0, 1].
        n_open: Initial number of open curves.
        n_closed: Initial number of closed curves.
        steps: Total optimization steps.
        prune_every: Prune/densify every N steps.
        prune_stop_before_end: Stop pruning this many steps before the end.
        lambda_xing: Weight for LIVE Xing loss. Only used when ``loss_config``
            is ``None`` (backward compatibility). When a ``LossConfig`` is
            provided, its ``lambda_xing`` takes precedence.
        log_every: Print loss every N steps.
        lr_step_size: StepLR decay period.
        lr_gamma: StepLR multiplicative decay factor.
        lr_scale: Global multiplier applied to all base learning rates.
        optimizer_type: ``"adam"`` (default) or ``"adan"``.
        prune_config: Pruning/densification heuristic thresholds. Uses
            defaults from ``PruneConfig()`` when ``None``.
        loss_config: Composite loss configuration. When ``None``, a default
            ``LossConfig`` is created with ``lambda_xing`` set from the
            ``lambda_xing`` parameter and all regularizers disabled to
            preserve backward-compatible behavior.
        callback: Optional callback(step, loss, scene) for monitoring.
            If the callback returns ``False``, training stops early.
        debug: When truthy, activate debug logging. When a string, use it as
            the output directory (default ``"debug_output"``).

    Returns:
        Optimized VectorGraphicsScene.
    """
    _, H, W = target.shape
    device = target.device
    if prune_config is None:
        prune_config = PruneConfig()

    # Build loss config: honor explicit loss_config, otherwise backward-compat
    if loss_config is None:
        loss_config = LossConfig(
            lambda_xing=lambda_xing,
            apply_shape_reg=False,
            apply_opacity_prior=False,
            apply_curvature=False,
            apply_boundary=False,
        )

    if debug:
        from .debug import (
            DebugTracker,
            check_health,
            collect_curve_stats,
            collect_gradient_stats,
            save_checkpoint,
            snapshot_scene,
        )
        from .metrics import compute_psnr

        output_dir = debug if isinstance(debug, str) else "debug_output"
        tracker = DebugTracker(
            run_name=f"fit_{H}x{W}",
            output_dir=output_dir,
            config={
                "n_open": n_open,
                "n_closed": n_closed,
                "steps": steps,
                "prune_every": prune_every,
                "prune_stop_before_end": prune_stop_before_end,
                "lambda_xing": loss_config.lambda_xing,
                "lr_step_size": lr_step_size,
                "lr_gamma": lr_gamma,
                "H": H,
                "W": W,
            },
        )
        health_history: dict = {}

    scene = VectorGraphicsScene(
        n_open=n_open, n_closed=n_closed, H=H, W=W,
    ).to(device)

    param_groups = _build_param_groups(scene, H, W, lr_scale=lr_scale)
    optimizer = _build_optimizer(param_groups, optimizer_type)

    lr_decay = 1.0  # accumulated decay factor

    for step in range(steps):
        # StepLR-style decay
        if step > 0 and step % lr_step_size == 0:
            lr_decay *= lr_gamma
            for group in optimizer.param_groups:
                group["lr"] *= lr_gamma

        scene.iter = step

        optimizer.zero_grad()

        rendered = scene(H, W)
        loss, loss_dict = compute_loss(rendered, target, scene, loss_config, step)
        loss.backward()

        if debug:
            grad_stats = collect_gradient_stats(scene)

        optimizer.step()

        # Defer loss.item() GPU-CPU sync to when we actually need the value
        need_loss_val = debug or callback is not None or step % log_every == 0
        loss_val = loss.item() if need_loss_val else 0.0

        if debug:
            psnr_val = compute_psnr(rendered.detach(), target).item()
            log_dict = {
                "loss": loss_val,
                "psnr": psnr_val,
                "n_open": scene.n_open,
                "n_closed": scene.n_closed,
                "mean_grad_norm": grad_stats["summary"]["mean_grad_norm"],
            }
            # Include all individual loss terms from loss_dict
            for k, v in loss_dict.items():
                if k != "total":
                    log_dict[f"loss_{k}"] = v
            tracker.log_scalars(step, log_dict)

            warnings = check_health(scene, step, health_history)
            if warnings:
                for w in warnings:
                    print(f"  [DEBUG WARNING] {w}")

            if step % 100 == 0:
                curve_stats = collect_curve_stats(scene, H, W)
                tracker.log_snapshot(step, "curve_stats", curve_stats)
                tracker.log_snapshot(step, "grad_stats", grad_stats)

            if step % 500 == 0 or step == steps - 1:
                save_checkpoint(scene, step, {"loss": loss_val, "psnr": psnr_val}, Path(output_dir))

        if step % log_every == 0:
            recon_val = loss_dict.get("reconstruction", loss_val)
            print(f"Step {step:5d}/{steps} | loss={loss_val:.6f} | recon={recon_val:.6f}")

        if callback is not None:
            if callback(step, loss_val, scene) is False:
                break

        # Pruning and densification
        should_prune = (
            step > 0
            and step % prune_every == 0
            and step < steps - prune_stop_before_end
        )
        if should_prune:
            if debug:
                pre_prune = snapshot_scene(scene)

            with torch.no_grad():
                needs_rebuild = _prune_and_densify(
                    scene, target, rendered, step, steps, H, W, prune_config, optimizer,
                )

            if debug:
                post_prune = snapshot_scene(scene)
                tracker.log_snapshot(step, "prune_before", pre_prune)
                tracker.log_snapshot(step, "prune_after", post_prune)
                with torch.no_grad():
                    rendered_at_prune = scene(H, W)
                tracker.log_snapshot(step, "rendered_at_prune", {"image": rendered_at_prune.detach().cpu()})

            # Only rebuild optimizer when topology change introduced new param groups
            # (e.g. 0→n transition). State surgery handles normal prune/densify.
            if needs_rebuild:
                param_groups = _build_param_groups(scene, H, W, lr_scale=lr_scale)
                optimizer = _build_optimizer(param_groups, optimizer_type)
                for group in optimizer.param_groups:
                    group["lr"] *= lr_decay

    if debug:
        tracker.finish()

    return scene


def _build_param_groups(
    scene: VectorGraphicsScene, H: int, W: int, lr_scale: float = 1.0,
) -> list[dict]:
    """Build optimizer parameter groups with per-type learning rates.

    Control points are in [-1, 1] normalized coordinates. The CP learning rate
    is scaled by resolution so the effective pixel displacement is ~0.25 px/iter
    regardless of image size. All base LRs are multiplied by ``lr_scale``.
    """
    cp_lr = 0.25 / max(H, W) * lr_scale
    groups: list[dict] = []

    if scene.n_open > 0:
        groups.extend([
            {"params": [scene.open_control_points], "lr": cp_lr, "name": "open_cp"},
            {"params": [scene.open_colors], "lr": 0.01 * lr_scale, "name": "open_colors"},
            {"params": [scene.open_opacities], "lr": 0.1 * lr_scale, "name": "open_opacities"},
            {"params": [scene.open_stroke_widths], "lr": 0.05 * lr_scale, "name": "open_stroke_widths"},
        ])

    if scene.n_closed > 0:
        groups.extend([
            {"params": [scene.closed_shared_pts], "lr": cp_lr, "name": "closed_shared_pts"},
            {"params": [scene.closed_interior_cp], "lr": cp_lr, "name": "closed_interior_cp"},
            {"params": [scene.closed_colors], "lr": 0.01 * lr_scale, "name": "closed_colors"},
            {"params": [scene.closed_opacities], "lr": 0.1 * lr_scale, "name": "closed_opacities"},
        ])

    # Depth parameter (mostly overwritten heuristically, lr=0 so optimizer
    # doesn't fight the overwrites but it still participates in state_dict)
    total_curves = scene.n_open + scene.n_closed
    if total_curves > 0:
        groups.append(
            {"params": [scene._depth], "lr": 0.0, "name": "depth"},
        )

    return groups


# ── Pruning & Densification ─────────────────────────────────────────────


def _prune_and_densify(
    scene: VectorGraphicsScene,
    target: Float[Tensor, "3 H W"],
    rendered: Float[Tensor, "3 H W"],
    step: int,
    total_steps: int,
    H: int,
    W: int,
    config: PruneConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> bool:
    """Prune low-quality curves and densify at high-error regions.

    Delegates mask computation to ``topology.py`` and applies the masks
    here via nn.Parameter assignment. When ``optimizer`` is provided,
    performs state surgery to preserve momentum for surviving curves
    instead of requiring a full optimizer rebuild.

    Strategy (paper Algorithm 1):
        1. Compute per-curve keep masks using topology heuristics
        2. Apply masks -- remove pruned curves by slicing parameters
        3. Find high-error regions and insert new curves

    Args:
        scene: Scene to modify in-place.
        target: Target image (3, H, W).
        rendered: Current rendered image (3, H, W).
        step: Current optimization step.
        total_steps: Total optimization steps.
        H, W: Image dimensions.
        config: Pruning heuristic thresholds.
        optimizer: Optimizer to perform state surgery on. When ``None``,
            no state surgery is performed (caller must rebuild).

    Returns:
        ``True`` if the optimizer needs a full rebuild (0->n transition
        introduced new param groups), ``False`` otherwise.
    """
    device = target.device
    if config is None:
        config = PruneConfig()

    progress = step / total_steps
    had_open = scene.n_open > 0
    had_closed = scene.n_closed > 0

    # ── Prune open curves ──
    pruned_open = 0
    open_keep_mask: Tensor | None = None
    if scene.n_open > 0:
        open_keep_mask, _open_metrics = compute_prune_mask_open(scene, progress, config, H, W)
        pruned_open = (~open_keep_mask).sum().item()

        if pruned_open > 0:
            if open_keep_mask.any():
                for attr in ("open_control_points", "open_colors", "open_opacities", "open_stroke_widths"):
                    old_param = getattr(scene, attr)
                    new_param = nn.Parameter(old_param[open_keep_mask].clone())
                    if optimizer is not None:
                        _prune_optimizer_state(optimizer, old_param, new_param, open_keep_mask)
                    setattr(scene, attr, new_param)
                scene.n_open = open_keep_mask.sum().item()
            else:
                for attr, shape in [
                    ("open_control_points", (0, 10, 2)),
                    ("open_colors", (0, 3)),
                    ("open_opacities", (0, 3)),
                    ("open_stroke_widths", (0,)),
                ]:
                    old_param = getattr(scene, attr)
                    new_param = nn.Parameter(torch.empty(*shape, device=device))
                    if optimizer is not None:
                        # Remove state for the dead parameter
                        if old_param in optimizer.state:
                            del optimizer.state[old_param]
                        for group in optimizer.param_groups:
                            for i, p in enumerate(group["params"]):
                                if p is old_param:
                                    group["params"][i] = new_param
                    setattr(scene, attr, new_param)
                scene.n_open = 0

    # ── Prune closed curves ──
    pruned_closed = 0
    closed_keep_mask: Tensor | None = None
    if scene.n_closed > 0:
        closed_keep_mask, _closed_metrics = compute_prune_mask_closed(scene, progress, config, H, W)
        pruned_closed = (~closed_keep_mask).sum().item()

        if pruned_closed > 0:
            if closed_keep_mask.any():
                for attr in ("closed_shared_pts", "closed_interior_cp", "closed_colors", "closed_opacities"):
                    old_param = getattr(scene, attr)
                    new_param = nn.Parameter(old_param[closed_keep_mask].clone())
                    if optimizer is not None:
                        _prune_optimizer_state(optimizer, old_param, new_param, closed_keep_mask)
                    setattr(scene, attr, new_param)
                scene.n_closed = closed_keep_mask.sum().item()
            else:
                num_interior = scene.closed_interior_cp.shape[2]
                for attr, shape in [
                    ("closed_shared_pts", (0, 2, 2)),
                    ("closed_interior_cp", (0, 2, num_interior, 2)),
                    ("closed_colors", (0, 3)),
                    ("closed_opacities", (0,)),
                ]:
                    old_param = getattr(scene, attr)
                    new_param = nn.Parameter(torch.empty(*shape, device=device))
                    if optimizer is not None:
                        if old_param in optimizer.state:
                            del optimizer.state[old_param]
                        for group in optimizer.param_groups:
                            for i, p in enumerate(group["params"]):
                                if p is old_param:
                                    group["params"][i] = new_param
                    setattr(scene, attr, new_param)
                scene.n_closed = 0

    # ── Rebuild _depth to match surviving curves ──
    if pruned_open > 0 or pruned_closed > 0:
        old_n_open = open_keep_mask.shape[0] if open_keep_mask is not None else 0
        old_n_closed = closed_keep_mask.shape[0] if closed_keep_mask is not None else 0
        depth_keep_parts: list[Tensor] = []
        if old_n_open > 0:
            depth_keep_parts.append(open_keep_mask)
        if old_n_closed > 0:
            depth_keep_parts.append(closed_keep_mask)
        if depth_keep_parts:
            depth_keep = torch.cat(depth_keep_parts)
            if depth_keep.any():
                old_depth = scene._depth
                new_depth = nn.Parameter(old_depth[depth_keep].clone())
                if optimizer is not None:
                    _prune_optimizer_state(optimizer, old_depth, new_depth, depth_keep)
                scene._depth = new_depth
            else:
                old_depth = scene._depth
                new_depth = nn.Parameter(torch.empty(0, 1, device=device))
                if optimizer is not None:
                    if old_depth in optimizer.state:
                        del optimizer.state[old_depth]
                    for group in optimizer.param_groups:
                        for i, p in enumerate(group["params"]):
                            if p is old_depth:
                                group["params"][i] = new_depth
                scene._depth = new_depth
        else:
            old_depth = scene._depth
            new_depth = nn.Parameter(torch.empty(0, 1, device=device))
            if optimizer is not None:
                if old_depth in optimizer.state:
                    del optimizer.state[old_depth]
                for group in optimizer.param_groups:
                    for i, p in enumerate(group["params"]):
                        if p is old_depth:
                            group["params"][i] = new_depth
            scene._depth = new_depth

    total_pruned = pruned_open + pruned_closed

    # ── Densification ──
    if total_pruned > 0:
        centers = compute_densify_centers(rendered, target, total_pruned, H, W)
        if centers.shape[0] > 0:
            error_map = (rendered - target).abs().mean(dim=0)  # (H, W)
            _densify_curves(scene, error_map, target, total_pruned, H, W, device, optimizer)

    # Check if we transitioned from 0→n curves (new param groups needed)
    now_has_open = scene.n_open > 0
    now_has_closed = scene.n_closed > 0
    new_open_group = not had_open and now_has_open
    new_closed_group = not had_closed and now_has_closed
    return new_open_group or new_closed_group


def _densify_curves(
    scene: VectorGraphicsScene,
    error_map: Float[Tensor, "H W"],
    target: Float[Tensor, "3 H W"],
    n_new: int,
    H: int,
    W: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
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

            # Aspect ratio of high-error sub-region -> determines curve type
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

    closed_cp_count = scene.closed_interior_cp.shape[2] + 2

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

        # High aspect -> edge-like -> open curve, else closed curve
        if aspect > 2.0 or scene.n_closed == 0:
            _make_open_curve(cx_px, cy_px, H, W, color, device, new_open_cps, new_open_colors)
        else:
            _make_closed_curve(
                cx_px, cy_px, H, W, closed_cp_count, color, device,
                new_closed_cps, new_closed_colors,
            )

    # ── Append new open curves ──
    n_new_open = len(new_open_cps)
    if new_open_cps:
        new_cps_t = torch.stack(new_open_cps)
        new_colors_t = torch.stack(new_open_colors)
        m = n_new_open
        new_opacities = torch.zeros(m, 3, device=device)
        new_stroke_widths = torch.zeros(m, device=device)

        if scene.n_open > 0:
            for attr, new_data in [
                ("open_control_points", new_cps_t),
                ("open_colors", new_colors_t),
                ("open_opacities", new_opacities),
                ("open_stroke_widths", new_stroke_widths),
            ]:
                old_param = getattr(scene, attr)
                new_param = nn.Parameter(torch.cat([old_param, new_data]))
                if optimizer is not None:
                    _extend_optimizer_state(optimizer, old_param, new_param, m)
                setattr(scene, attr, new_param)
        else:
            scene.open_control_points = nn.Parameter(new_cps_t)
            scene.open_colors = nn.Parameter(new_colors_t)
            scene.open_opacities = nn.Parameter(new_opacities)
            scene.open_stroke_widths = nn.Parameter(new_stroke_widths)
        scene.n_open = scene.open_control_points.shape[0]

    # ── Append new closed curves ──
    n_new_closed = len(new_closed_cps)
    if new_closed_cps:
        new_bcp_t = torch.stack(new_closed_cps)
        new_colors_t = torch.stack(new_closed_colors)
        m = n_new_closed
        new_opacities = torch.zeros(m, device=device)

        if scene.n_closed > 0:
            new_shared = torch.stack([new_bcp_t[:, 0, 0], new_bcp_t[:, 0, -1]], dim=1)  # (M, 2, 2)
            new_interior = new_bcp_t[:, :, 1:-1, :]  # (M, 2, num_cp-2, 2)
            for attr, new_data in [
                ("closed_shared_pts", new_shared),
                ("closed_interior_cp", new_interior),
                ("closed_colors", new_colors_t),
                ("closed_opacities", new_opacities),
            ]:
                old_param = getattr(scene, attr)
                new_param = nn.Parameter(torch.cat([old_param, new_data]))
                if optimizer is not None:
                    _extend_optimizer_state(optimizer, old_param, new_param, m)
                setattr(scene, attr, new_param)
        else:
            new_shared = torch.stack([new_bcp_t[:, 0, 0], new_bcp_t[:, 0, -1]], dim=1)
            new_interior = new_bcp_t[:, :, 1:-1, :]
            scene.closed_shared_pts = nn.Parameter(new_shared)
            scene.closed_interior_cp = nn.Parameter(new_interior)
            scene.closed_colors = nn.Parameter(new_colors_t)
            scene.closed_opacities = nn.Parameter(new_opacities)
        scene.n_closed = scene.closed_shared_pts.shape[0]

    # ── Splice depth entries for new curves ──
    total_new = n_new_open + n_new_closed
    if total_new > 0:
        # Layout: [existing_open | new_open | existing_closed | new_closed]
        old_depth_param = scene._depth
        old_depth = old_depth_param.data
        old_n_open = scene.n_open - n_new_open  # n_open already updated above
        old_n_closed = scene.n_closed - n_new_closed
        parts: list[Tensor] = []
        if old_n_open > 0:
            parts.append(old_depth[:old_n_open])
        if n_new_open > 0:
            parts.append(torch.ones(n_new_open, 1, device=device))
        if old_n_closed > 0:
            parts.append(old_depth[old_n_open:old_n_open + old_n_closed])
        if n_new_closed > 0:
            parts.append(torch.ones(n_new_closed, 1, device=device))
        if parts:
            new_depth_param = nn.Parameter(torch.cat(parts, dim=0))
            if optimizer is not None:
                # Depth splice: insert new_open entries after existing open,
                # then append new_closed at the end. Use general splice for
                # the open insertion and extend for the closed append.
                # Simpler: build state manually to match the new layout.
                state = optimizer.state.get(old_depth_param)
                if state is not None:
                    new_state: dict = {"step": state.get("step", 0)}
                    for key, val in state.items():
                        if key == "step":
                            continue
                        if isinstance(val, Tensor):
                            # Same splice as the data: [old_open | zeros_open | old_closed | zeros_closed]
                            s_parts: list[Tensor] = []
                            if old_n_open > 0:
                                s_parts.append(val[:old_n_open])
                            if n_new_open > 0:
                                s_parts.append(torch.zeros(n_new_open, *val.shape[1:], device=val.device, dtype=val.dtype))
                            if old_n_closed > 0:
                                s_parts.append(val[old_n_open:old_n_open + old_n_closed])
                            if n_new_closed > 0:
                                s_parts.append(torch.zeros(n_new_closed, *val.shape[1:], device=val.device, dtype=val.dtype))
                            if s_parts:
                                new_state[key] = torch.cat(s_parts, dim=0)
                        else:
                            new_state[key] = val
                    del optimizer.state[old_depth_param]
                    optimizer.state[new_depth_param] = new_state
                    for group in optimizer.param_groups:
                        for i, p in enumerate(group["params"]):
                            if p is old_depth_param:
                                group["params"][i] = new_depth_param
                                break
            scene._depth = new_depth_param


def _make_open_curve(
    cx_px: float, cy_px: float,
    H: int, W: int,
    color: Float[Tensor, " 3"], device: torch.device,
    out_cps: list[Tensor], out_colors: list[Tensor],
) -> None:
    """Create one new open curve in [-1, 1] coords near a pixel center."""
    center_px = torch.tensor([[cx_px, cy_px]], device=device)
    center_n = pixel_to_model(center_px, H, W).squeeze(0)  # (2,)
    spread = 0.1  # doubled from 0.05 for [-1, 1] range

    cp = torch.zeros(10, 2, device=device)
    t_vals = torch.linspace(-1, 1, 10, device=device)
    cp[:, 0] = center_n[0] + t_vals * spread + torch.randn(10, device=device) * spread * 0.3
    cp[:, 1] = center_n[1] + torch.randn(10, device=device) * spread * 0.5
    cp = cp.clamp(-1, 1)

    out_cps.append(cp)
    out_colors.append(color)


def _make_closed_curve(
    cx_px: float, cy_px: float,
    H: int, W: int,
    num_cp: int, color: Float[Tensor, " 3"], device: torch.device,
    out_cps: list[Tensor], out_colors: list[Tensor],
) -> None:
    """Create one new closed curve in [-1, 1] coords near a pixel center."""
    center_px = torch.tensor([[cx_px, cy_px]], device=device)
    center_n = pixel_to_model(center_px, H, W).squeeze(0)  # (2,)
    size = 0.08  # doubled from 0.04 for [-1, 1] range

    bcp = torch.zeros(2, num_cp, 2, device=device)
    t = torch.linspace(0, 1, num_cp, device=device)
    for b in range(2):
        y_off = size * (1 if b == 0 else -1)
        bcp[b, :, 0] = center_n[0] + (t - 0.5) * size * 2
        bcp[b, :, 1] = center_n[1] + y_off + torch.randn(num_cp, device=device) * size * 0.3

    # Shared endpoints
    shared_start = (bcp[0, 0] + bcp[1, 0]) / 2
    shared_end = (bcp[0, -1] + bcp[1, -1]) / 2
    bcp[0, 0] = shared_start
    bcp[1, 0] = shared_start
    bcp[0, -1] = shared_end
    bcp[1, -1] = shared_end
    bcp = bcp.clamp(-1, 1)

    out_cps.append(bcp)
    out_colors.append(color)
