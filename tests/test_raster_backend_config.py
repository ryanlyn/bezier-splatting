"""Backend configuration plumbing tests."""

import warnings
from unittest.mock import patch

import pytest
import torch

from bezier_splatting.model import VectorGraphicsScene
from bezier_splatting.optimization import fit_image
from bezier_splatting.rasterizer import _resolve_backend
from bezier_splatting.sampling import GaussianParams


def test_scene_forward_uses_configured_raster_backend_args():
    scene = VectorGraphicsScene(
        n_open=1,
        n_closed=0,
        H=32,
        W=32,
        samples_per_open=4,
        raster_backend="pytorch",
        raster_tile_size=24,
        raster_chunk_size=80,
    )

    with patch("bezier_splatting.model.rasterize", return_value=torch.ones(3, 32, 32)) as mocked:
        out = scene(32, 32)

    assert out.shape == (3, 32, 32)
    _, _, _ = mocked.call_args.args[:3]
    assert mocked.call_args.kwargs["backend"] == "pytorch"
    assert mocked.call_args.kwargs["tile_size"] == 24
    assert mocked.call_args.kwargs["chunk_size"] == 80


def test_scene_forward_allows_per_call_backend_overrides():
    scene = VectorGraphicsScene(
        n_open=1,
        n_closed=0,
        H=32,
        W=32,
        samples_per_open=4,
        raster_backend="pytorch",
        raster_tile_size=24,
        raster_chunk_size=80,
    )

    with patch("bezier_splatting.model.rasterize", return_value=torch.ones(3, 32, 32)) as mocked:
        _ = scene(32, 32, backend="reference", tile_size=12, chunk_size=7)

    assert mocked.call_args.kwargs["backend"] == "reference"
    assert mocked.call_args.kwargs["tile_size"] == 12
    assert mocked.call_args.kwargs["chunk_size"] == 7


def test_fit_image_propagates_raster_backend_config():
    target = torch.rand(3, 16, 16)

    scene = fit_image(
        target,
        n_open=1,
        n_closed=0,
        steps=0,
        raster_backend="pytorch",
        raster_tile_size=20,
        raster_chunk_size=72,
    )

    assert scene.raster_backend == "pytorch"
    assert scene.raster_tile_size == 20
    assert scene.raster_chunk_size == 72


def test_legacy_backend_aliases_resolve_to_pytorch():
    """'reference' and 'mps' are accepted as aliases for 'pytorch'."""
    cpu = torch.device("cpu")
    assert _resolve_backend("reference", cpu) == "pytorch"
    assert _resolve_backend("mps", cpu) == "pytorch"
    assert _resolve_backend("pytorch", cpu) == "pytorch"


def test_gsplat_on_cpu_raises_clear_error():
    """Requesting gsplat on a CPU device raises ValueError."""
    cpu = torch.device("cpu")
    with pytest.raises(ValueError, match="gsplat backend requires a CUDA device"):
        _resolve_backend("gsplat", cpu)


def test_gsplat_not_installed_raises_clear_error():
    """Requesting gsplat when the library is missing raises ImportError."""
    cuda = torch.device("cuda")
    with patch("bezier_splatting.rasterizer._check_gsplat", return_value=False):
        with pytest.raises(ImportError, match="gsplat.*not installed"):
            _resolve_backend("gsplat", cuda)


def test_auto_backend_on_cuda_requires_gsplat():
    """Auto backend raises when CUDA is selected but gsplat is unavailable."""
    cuda = torch.device("cuda")
    with patch("bezier_splatting.rasterizer._check_gsplat", return_value=False):
        with pytest.raises(ImportError, match="`auto` on CUDA requires `gsplat`"):
            _resolve_backend("auto", cuda)


def test_auto_backend_on_cuda_no_warning_when_gsplat_available():
    """Auto backend resolves to gsplat silently when gsplat is available."""
    cuda = torch.device("cuda")
    with patch("bezier_splatting.rasterizer._check_gsplat", return_value=True):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            resolved = _resolve_backend("auto", cuda)
    assert resolved == "gsplat"
    assert len(record) == 0


def test_explicit_pytorch_backend_on_cuda_warns_and_resolves():
    """Explicit pytorch backend on CUDA should warn but still proceed."""
    cuda = torch.device("cuda")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        resolved = _resolve_backend("pytorch", cuda)
    assert resolved == "pytorch"
    assert any("selected explicitly on CUDA" in str(w.message) for w in record)


def test_unknown_backend_raises_value_error():
    """An unrecognized backend string raises ValueError."""
    cpu = torch.device("cpu")
    with pytest.raises(ValueError, match="Unknown raster backend"):
        _resolve_backend("bogus", cpu)
