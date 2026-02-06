"""Parity checks for the vectorized MPS rasterizer backend."""

import torch
import torch.nn.functional as F
import pytest

from bezier_splatting.rasterizer import rasterize
from bezier_splatting.sampling import GaussianParams


def _make_gaussians(
    n: int,
    H: int,
    W: int,
    device: torch.device,
    requires_grad: bool = False,
) -> GaussianParams:
    scale = torch.tensor([W, H], device=device, dtype=torch.float32)

    means = torch.rand(n, 2, device=device) * scale
    scales = torch.rand(n, 2, device=device) * 3.0 + 0.8
    rotations = torch.rand(n, device=device) * (2.0 * torch.pi)
    colors = torch.rand(n, 3, device=device)
    opacities = torch.randn(n, device=device)

    if requires_grad:
        means.requires_grad_(True)
        scales.requires_grad_(True)
        rotations.requires_grad_(True)
        colors.requires_grad_(True)
        opacities.requires_grad_(True)

    return GaussianParams(
        means=means,
        scales=scales,
        rotations=rotations,
        colors=colors,
        opacities=opacities,
        curve_ids=torch.arange(n, device=device, dtype=torch.long),
    )


def _to_device(g: GaussianParams, device: torch.device) -> GaussianParams:
    return GaussianParams(
        means=g.means.to(device),
        scales=g.scales.to(device),
        rotations=g.rotations.to(device),
        colors=g.colors.to(device),
        opacities=g.opacities.to(device),
        curve_ids=g.curve_ids.to(device),
    )


def test_mps_backend_matches_reference_on_cpu():
    torch.manual_seed(0)
    H = W = 64
    g = _make_gaussians(128, H, W, device=torch.device("cpu"))

    ref = rasterize(g, H, W, backend="reference", tile_size=16, chunk_size=16)
    out = rasterize(g, H, W, backend="mps", tile_size=16, chunk_size=16)

    assert torch.allclose(out, ref, atol=2e-5, rtol=1e-4)


def test_mps_backend_gradients_match_reference_on_cpu():
    torch.manual_seed(1)
    H = W = 48
    target = torch.rand(3, H, W)

    g_ref = _make_gaussians(72, H, W, device=torch.device("cpu"), requires_grad=True)
    out_ref = rasterize(g_ref, H, W, backend="reference", tile_size=16, chunk_size=16)
    loss_ref = F.mse_loss(out_ref, target)
    loss_ref.backward()

    ref_means_grad = g_ref.means.grad.detach().clone()
    ref_opacity_grad = g_ref.opacities.grad.detach().clone()

    g_mps = GaussianParams(
        means=g_ref.means.detach().clone().requires_grad_(True),
        scales=g_ref.scales.detach().clone().requires_grad_(True),
        rotations=g_ref.rotations.detach().clone().requires_grad_(True),
        colors=g_ref.colors.detach().clone().requires_grad_(True),
        opacities=g_ref.opacities.detach().clone().requires_grad_(True),
        curve_ids=g_ref.curve_ids.detach().clone(),
    )
    out_mps = rasterize(g_mps, H, W, backend="mps", tile_size=16, chunk_size=16)
    loss_mps = F.mse_loss(out_mps, target)
    loss_mps.backward()

    assert torch.allclose(out_mps, out_ref.detach(), atol=2e-5, rtol=1e-4)
    assert torch.allclose(g_mps.means.grad, ref_means_grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(g_mps.opacities.grad, ref_opacity_grad, atol=1e-4, rtol=1e-3)


def test_auto_backend_uses_reference_on_cpu():
    torch.manual_seed(3)
    H = W = 40
    g = _make_gaussians(50, H, W, device=torch.device("cpu"))

    auto = rasterize(g, H, W, backend="auto")
    ref = rasterize(g, H, W, backend="reference")

    assert torch.allclose(auto, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_auto_backend_runs_on_mps_and_matches_cpu_reference():
    torch.manual_seed(5)
    H = W = 64
    g_cpu = _make_gaussians(96, H, W, device=torch.device("cpu"))

    ref = rasterize(g_cpu, H, W, backend="reference")

    g_mps = _to_device(g_cpu, torch.device("mps"))
    out = rasterize(g_mps, H, W, backend="auto").cpu()

    assert torch.allclose(out, ref, atol=6e-3, rtol=2e-3)
