"""Data collectors for gradient statistics, curve stats, and scene snapshots."""

from __future__ import annotations

import torch
from torch import Tensor

from ..model import VectorGraphicsScene


def collect_gradient_stats(scene: VectorGraphicsScene) -> dict:
    """Read .grad from all parameters after backward(). Return gradient info.

    Returns:
        Dict with keys:
            open_cp_grad: Tensor | None — (N, 10, 2) gradient magnitudes
            closed_cp_grad: Tensor | None — (N, 2, num_cp, 2) gradient magnitudes
            open_opacity_grad: Tensor | None — (N, 3) gradient magnitudes
            closed_opacity_grad: Tensor | None — (N,) gradient magnitudes
            summary: dict with mean_grad_norm, max_grad_norm, has_nan
    """
    result: dict = {
        "open_cp_grad": None,
        "closed_cp_grad": None,
        "open_opacity_grad": None,
        "closed_opacity_grad": None,
    }

    all_grads: list[Tensor] = []

    if scene.n_open > 0:
        if scene.open_control_points.grad is not None:
            g = scene.open_control_points.grad.detach()
            result["open_cp_grad"] = g.abs()
            all_grads.append(g.reshape(-1))
        if scene.open_opacities.grad is not None:
            g = scene.open_opacities.grad.detach()
            result["open_opacity_grad"] = g.abs()
            all_grads.append(g.reshape(-1))

    if scene.n_closed > 0:
        if scene.closed_boundary_cp.grad is not None:
            g = scene.closed_boundary_cp.grad.detach()
            result["closed_cp_grad"] = g.abs()
            all_grads.append(g.reshape(-1))
        if scene.closed_opacities.grad is not None:
            g = scene.closed_opacities.grad.detach()
            result["closed_opacity_grad"] = g.abs()
            all_grads.append(g.reshape(-1))

    if all_grads:
        flat = torch.cat(all_grads)
        result["summary"] = {
            "mean_grad_norm": flat.abs().mean().item(),
            "max_grad_norm": flat.abs().max().item(),
            "has_nan": bool(torch.isnan(flat).any().item()),
        }
    else:
        result["summary"] = {
            "mean_grad_norm": 0.0,
            "max_grad_norm": 0.0,
            "has_nan": False,
        }

    return result


def collect_curve_stats(scene: VectorGraphicsScene, H: int, W: int) -> dict:
    """Extract per-curve stats in display space.

    Returns:
        Dict with keys:
            open_opacities: Tensor (N, 3) in sigmoid space
            open_widths: Tensor (N,) in pixel space
            closed_opacities: Tensor (N,) in sigmoid space
            mean_scales: dict with 'open' and 'closed' Tensors (per-curve mean sigma)
            n_open: int
            n_closed: int
    """
    device = (
        scene.open_control_points.device
        if scene.n_open > 0
        else scene.closed_boundary_cp.device
    )

    result: dict = {
        "open_opacities": torch.empty(0, 3, device=device),
        "open_widths": torch.empty(0, device=device),
        "closed_opacities": torch.empty(0, device=device),
        "mean_scales": {"open": torch.empty(0, device=device), "closed": torch.empty(0, device=device)},
        "n_open": scene.n_open,
        "n_closed": scene.n_closed,
    }

    if scene.n_open > 0:
        result["open_opacities"] = torch.sigmoid(scene.open_opacities).detach()
        result["open_widths"] = (
            0.5 + torch.sigmoid(scene.open_stroke_widths).detach() * 4.5
        )

        # Mean scale: sample the curves and average sigma across samples
        with torch.no_grad():
            from ..sampling import OpenCurveSampler

            sampler = scene.open_sampler
            gp = sampler(
                scene.open_control_points,
                torch.sigmoid(scene.open_colors),
                scene.open_opacities,
                scene.open_stroke_widths,
                H, W,
            )
            if gp.means.shape[0] > 0:
                K = sampler.samples_per_curve
                # scales shape: (N*K, 2) -> (N, K, 2)
                scales = gp.scales.reshape(scene.n_open, K, 2)
                result["mean_scales"]["open"] = scales.mean(dim=(1, 2))  # (N,)

    if scene.n_closed > 0:
        result["closed_opacities"] = torch.sigmoid(scene.closed_opacities).detach()

        with torch.no_grad():
            sampler = scene.closed_sampler
            bcp = scene._enforce_shared_endpoints()
            gp = sampler(
                bcp,
                torch.sigmoid(scene.closed_colors),
                scene.closed_opacities,
                H, W,
            )
            if gp.means.shape[0] > 0:
                R_total = sampler.num_intermediate + 2
                K = sampler.samples_per_curve
                # scales: (N*R_total*K, 2) -> (N, R_total*K, 2)
                scales = gp.scales.reshape(scene.n_closed, R_total * K, 2)
                result["mean_scales"]["closed"] = scales.mean(dim=(1, 2))  # (N,)

    return result


def snapshot_scene(scene: VectorGraphicsScene) -> dict:
    """Deep-copy all parameter tensors (detached, CPU). For before/after comparison."""
    snap: dict = {}
    for name, param in scene.named_parameters():
        snap[name] = param.detach().cpu().clone()
    for name, buf in scene.named_buffers():
        snap[name] = buf.detach().cpu().clone()
    snap["n_open"] = scene.n_open
    snap["n_closed"] = scene.n_closed
    return snap
