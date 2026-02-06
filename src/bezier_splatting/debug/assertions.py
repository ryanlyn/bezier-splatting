"""Runtime health checks for the optimization loop.

Detects NaN/Inf parameters, dead curves, and scale collapse early
so the user can diagnose divergence before it ruins a long run.
"""

import torch

from ..model import VectorGraphicsScene


def check_health(
    scene: VectorGraphicsScene,
    step: int,
    history: dict,
) -> list[str]:
    """Run all health checks on the scene. Return a list of warning strings.

    An empty list means the scene is healthy.

    Checks performed:
        1. NaN/Inf in any parameter or gradient.
        2. Dead curves: sigmoid(opacity) < 0.01 for 200+ consecutive steps.
        3. Scale collapse: any sigma < 0.15 (just above the 0.1 clamp floor).

    Args:
        scene: The current VectorGraphicsScene.
        step: Current optimization step number.
        history: A mutable dict the caller maintains across steps.
            This function reads/writes ``'opacity_history'`` and
            ``'dead_steps'`` keys.

    Returns:
        List of human-readable warning strings (empty = healthy).
    """
    warnings: list[str] = []

    # ── 1. NaN / Inf in parameters and gradients ──
    for name, param in scene.named_parameters():
        if torch.isnan(param).any():
            warnings.append(f"step {step}: NaN in parameter '{name}'")
        if torch.isinf(param).any():
            warnings.append(f"step {step}: Inf in parameter '{name}'")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                warnings.append(f"step {step}: NaN in gradient of '{name}'")
            if torch.isinf(param.grad).any():
                warnings.append(f"step {step}: Inf in gradient of '{name}'")

    # ── 2. Dead curves (opacity stuck below 0.01 for 200+ steps) ──
    _check_dead_curves(scene, step, history, warnings)

    # ── 3. Scale collapse ──
    _check_scale_collapse(scene, step, warnings)

    return warnings


def _check_dead_curves(
    scene: VectorGraphicsScene,
    step: int,
    history: dict,
    warnings: list[str],
) -> None:
    """Track per-curve max opacity across steps. Warn if dead for 200+ steps."""
    dead_threshold = 0.01
    dead_patience = 200

    if "dead_steps" not in history:
        history["dead_steps"] = {}

    dead_steps: dict[str, torch.Tensor] = history["dead_steps"]

    # Open curves: max across 3 segment opacities
    if scene.n_open > 0:
        open_op = torch.sigmoid(scene.open_opacities).detach()  # (N, 3)
        open_max = open_op.max(dim=-1).values  # (N,)
        alive = open_max >= dead_threshold

        key = "open"
        if key not in dead_steps or dead_steps[key].shape[0] != scene.n_open:
            dead_steps[key] = torch.zeros(scene.n_open, dtype=torch.long)

        dead_steps[key][alive] = 0
        dead_steps[key][~alive] += 1

        n_dead = (dead_steps[key] >= dead_patience).sum().item()
        if n_dead > 0:
            warnings.append(
                f"step {step}: {n_dead} open curve(s) dead "
                f"(opacity < {dead_threshold} for {dead_patience}+ steps)"
            )

    # Closed curves: scalar legacy or 3-value profile
    if scene.n_closed > 0:
        closed_op = torch.sigmoid(scene.closed_opacities).detach()
        if closed_op.ndim == 2:
            closed_alive_score = closed_op.max(dim=-1).values
        else:
            closed_alive_score = closed_op
        alive = closed_alive_score >= dead_threshold

        key = "closed"
        if key not in dead_steps or dead_steps[key].shape[0] != scene.n_closed:
            dead_steps[key] = torch.zeros(scene.n_closed, dtype=torch.long)

        dead_steps[key][alive] = 0
        dead_steps[key][~alive] += 1

        n_dead = (dead_steps[key] >= dead_patience).sum().item()
        if n_dead > 0:
            warnings.append(
                f"step {step}: {n_dead} closed curve(s) dead "
                f"(opacity < {dead_threshold} for {dead_patience}+ steps)"
            )


def _check_scale_collapse(
    scene: VectorGraphicsScene,
    step: int,
    warnings: list[str],
) -> None:
    """Warn if any stroke width or closed scale is near the 0.1 clamp floor."""
    collapse_threshold = 0.15

    if scene.n_open > 0:
        # Stroke widths in pixel space: 0.5 + sigmoid(w) * 4.5
        widths = 0.5 + torch.sigmoid(scene.open_stroke_widths).detach() * 4.5
        n_collapsed = (widths < collapse_threshold).sum().item()
        if n_collapsed > 0:
            warnings.append(
                f"step {step}: {n_collapsed} open curve(s) with "
                f"stroke width < {collapse_threshold}px (near clamp floor)"
            )
