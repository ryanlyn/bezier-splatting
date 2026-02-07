"""Parity checks across rasterizer backends."""

import torch
import torch.nn.functional as F
import pytest
from unittest.mock import patch

from bezier_splatting.rasterizer import rasterize, _check_gsplat
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


def _clone_gaussians(g: GaussianParams, requires_grad: bool = False) -> GaussianParams:
    return GaussianParams(
        means=g.means.detach().clone().requires_grad_(requires_grad),
        scales=g.scales.detach().clone().requires_grad_(requires_grad),
        rotations=g.rotations.detach().clone().requires_grad_(requires_grad),
        colors=g.colors.detach().clone().requires_grad_(requires_grad),
        opacities=g.opacities.detach().clone().requires_grad_(requires_grad),
        curve_ids=g.curve_ids.detach().clone(),
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


# --- PyTorch backend consistency ---


def test_pytorch_backend_consistency_on_cpu():
    """The pytorch backend produces consistent results on CPU."""
    torch.manual_seed(0)
    H = W = 64
    g = _make_gaussians(128, H, W, device=torch.device("cpu"))

    out1 = rasterize(g, H, W, backend="pytorch", tile_size=16, chunk_size=16)
    out2 = rasterize(g, H, W, backend="pytorch", tile_size=16, chunk_size=16)

    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)


def test_legacy_aliases_match_pytorch():
    """Legacy backend names 'reference' and 'mps' produce identical output to 'pytorch'."""
    torch.manual_seed(10)
    H = W = 48
    g = _make_gaussians(64, H, W, device=torch.device("cpu"))

    pytorch = rasterize(g, H, W, backend="pytorch", tile_size=16, chunk_size=16)
    ref = rasterize(g, H, W, backend="reference", tile_size=16, chunk_size=16)
    mps = rasterize(g, H, W, backend="mps", tile_size=16, chunk_size=16)

    assert torch.allclose(pytorch, ref, atol=0.0, rtol=0.0)
    assert torch.allclose(pytorch, mps, atol=0.0, rtol=0.0)


def test_pytorch_gradients_on_cpu():
    """Gradient flow through the pytorch backend."""
    torch.manual_seed(1)
    H = W = 48
    target = torch.rand(3, H, W)

    g = _make_gaussians(72, H, W, device=torch.device("cpu"), requires_grad=True)
    out = rasterize(g, H, W, backend="pytorch", tile_size=16, chunk_size=16)
    loss = F.mse_loss(out, target)
    loss.backward()

    assert g.means.grad is not None
    assert g.opacities.grad is not None
    assert g.means.grad.abs().sum() > 0
    assert g.opacities.grad.abs().sum() > 0


def test_auto_backend_uses_pytorch_on_cpu():
    """Auto backend resolves to pytorch on CPU (no gsplat)."""
    torch.manual_seed(3)
    H = W = 40
    g = _make_gaussians(50, H, W, device=torch.device("cpu"))

    auto = rasterize(g, H, W, backend="auto")
    pytorch = rasterize(g, H, W, backend="pytorch")

    assert torch.allclose(auto, pytorch, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_auto_backend_runs_on_mps_and_matches_cpu():
    """Auto on MPS device still uses pytorch backend, close to CPU result."""
    torch.manual_seed(5)
    H = W = 64
    g_cpu = _make_gaussians(96, H, W, device=torch.device("cpu"))

    ref = rasterize(g_cpu, H, W, backend="pytorch")

    g_mps = _to_device(g_cpu, torch.device("mps"))
    out = rasterize(g_mps, H, W, backend="auto").cpu()

    assert torch.allclose(out, ref, atol=6e-3, rtol=2e-3)


# --- gsplat backend (CUDA-conditional) ---


_has_cuda = torch.cuda.is_available()
_has_gsplat = _check_gsplat()
_skip_no_gsplat = pytest.mark.skipif(
    not (_has_cuda and _has_gsplat),
    reason="Requires CUDA device and gsplat library",
)


@_skip_no_gsplat
def test_gsplat_matches_pytorch():
    """gsplat backend produces numerically close results to pytorch backend."""
    torch.manual_seed(42)
    H = W = 64
    g_cpu = _make_gaussians(128, H, W, device=torch.device("cpu"))

    ref = rasterize(g_cpu, H, W, backend="pytorch", tile_size=16, chunk_size=16)

    g_cuda = _to_device(g_cpu, torch.device("cuda"))
    out = rasterize(g_cuda, H, W, backend="gsplat", tile_size=16).cpu()

    # Tolerance accounts for floating-point differences between CUDA and CPU
    # implementations plus gsplat's internal 1/255 alpha cutoff vs our 0.99 clamp.
    assert torch.allclose(out, ref, atol=0.015, rtol=1e-3)


@_skip_no_gsplat
def test_gsplat_gradients_exist():
    """Gradients flow through all Gaussian parameters in the gsplat backend."""
    torch.manual_seed(43)
    H = W = 48
    target = torch.rand(3, H, W, device="cuda")

    g = _make_gaussians(64, H, W, device=torch.device("cuda"), requires_grad=True)
    out = rasterize(g, H, W, backend="gsplat", tile_size=16)
    loss = F.mse_loss(out, target)
    loss.backward()

    for name in ("means", "scales", "rotations", "colors", "opacities"):
        grad = getattr(g, name).grad
        assert grad is not None, f"No gradient for {name}"
        assert grad.abs().sum() > 0, f"Zero gradient for {name}"


@_skip_no_gsplat
def test_gsplat_compositing_order():
    """Front-to-back compositing in gsplat matches expected behavior."""
    g = GaussianParams(
        means=torch.tensor([[16.0, 16.0], [16.0, 16.0]], device="cuda"),
        scales=torch.tensor([[2.0, 2.0], [10.0, 10.0]], device="cuda"),
        rotations=torch.tensor([0.0, 0.0], device="cuda"),
        colors=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device="cuda"),
        opacities=torch.tensor([5.0, 5.0], device="cuda"),
        curve_ids=torch.tensor([0, 1], device="cuda"),
    )
    img = rasterize(g, 32, 32, backend="gsplat")
    center = img[:, 16, 16]
    assert center[2] > center[0], (
        f"Front-to-back order broken: blue={center[2]:.3f} should > red={center[0]:.3f}"
    )


@_skip_no_gsplat
def test_gsplat_empty_scene():
    """gsplat backend handles empty Gaussians correctly."""
    g = GaussianParams(
        means=torch.empty(0, 2, device="cuda"),
        scales=torch.empty(0, 2, device="cuda"),
        rotations=torch.empty(0, device="cuda"),
        colors=torch.empty(0, 3, device="cuda"),
        opacities=torch.empty(0, device="cuda"),
        curve_ids=torch.empty(0, dtype=torch.long, device="cuda"),
    )
    img = rasterize(g, 32, 32, backend="gsplat")
    assert img.shape == (3, 32, 32)
    assert torch.allclose(img.cpu(), torch.ones(3, 32, 32), atol=1e-5)
