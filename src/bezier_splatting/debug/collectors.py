"""Data collectors for gradient statistics, curve stats, and scene snapshots."""

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
            closed_opacity_grad: Tensor | None — (N,) or (N, 3) gradient magnitudes
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
        # closed_boundary_cp is a computed property (not a leaf), so access
        # the actual leaf parameters for gradient information.
        shared_grad = scene.closed_shared_pts.grad
        interior_grad = scene.closed_interior_cp.grad
        if shared_grad is not None:
            all_grads.append(shared_grad.detach().reshape(-1))
        if interior_grad is not None:
            all_grads.append(interior_grad.detach().reshape(-1))
        if shared_grad is not None or interior_grad is not None:
            # Reconstruct full gradient shape for downstream viz
            bcp = scene._assemble_boundary_cp()
            # Use autograd-free magnitude estimate from the leaf grads
            result["closed_cp_grad"] = torch.zeros_like(bcp)
            if shared_grad is not None:
                result["closed_cp_grad"][:, :, 0, :] += shared_grad[:, 0, :].unsqueeze(1).abs()
                result["closed_cp_grad"][:, :, -1, :] += shared_grad[:, 1, :].unsqueeze(1).abs()
            if interior_grad is not None:
                result["closed_cp_grad"][:, :, 1:-1, :] = interior_grad.detach().abs()
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
            closed_opacities: Tensor (N, 3) in sigmoid space
            mean_scales: dict with 'open' and 'closed' Tensors (per-curve mean sigma)
            n_open: int
            n_closed: int
    """
    device = (
        scene.open_control_points.device
        if scene.n_open > 0
        else scene.closed_shared_pts.device
    )

    result: dict = {
        "open_opacities": torch.empty(0, 3, device=device),
        "open_widths": torch.empty(0, device=device),
        "closed_opacities": torch.empty(0, 3, device=device),
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
                scene.open_colors.clamp(0.0, 1.0),
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
            bcp = scene._assemble_boundary_cp()
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
    """Deep-copy parameters/buffers plus render metadata for scene reconstruction."""
    snap: dict = {}
    for name, param in scene.named_parameters():
        snap[name] = param.detach().cpu().clone()
    for name, buf in scene.named_buffers():
        snap[name] = buf.detach().cpu().clone()
    snap["n_open"] = scene.n_open
    snap["n_closed"] = scene.n_closed
    snap["samples_per_open"] = scene.open_sampler.samples_per_curve
    snap["samples_per_closed_curve"] = scene.closed_sampler.samples_per_curve
    snap["num_intermediate"] = scene.closed_sampler.num_intermediate
    snap["closed_sampling_mode"] = scene.closed_sampling_mode
    snap["raster_backend"] = scene.raster_backend
    snap["raster_tile_size"] = scene.raster_tile_size
    snap["raster_chunk_size"] = scene.raster_chunk_size
    return snap
