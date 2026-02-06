"""Backend configuration plumbing tests."""

from unittest.mock import patch

import torch

from bezier_splatting.model import VectorGraphicsScene
from bezier_splatting.optimization import fit_image


def test_scene_forward_uses_configured_raster_backend_args():
    scene = VectorGraphicsScene(
        n_open=1,
        n_closed=0,
        H=32,
        W=32,
        samples_per_open=4,
        raster_backend="mps",
        raster_tile_size=24,
        raster_chunk_size=80,
    )

    with patch("bezier_splatting.model.rasterize", return_value=torch.ones(3, 32, 32)) as mocked:
        out = scene(32, 32)

    assert out.shape == (3, 32, 32)
    _, _, _ = mocked.call_args.args[:3]
    assert mocked.call_args.kwargs["backend"] == "mps"
    assert mocked.call_args.kwargs["tile_size"] == 24
    assert mocked.call_args.kwargs["chunk_size"] == 80


def test_scene_forward_allows_per_call_backend_overrides():
    scene = VectorGraphicsScene(
        n_open=1,
        n_closed=0,
        H=32,
        W=32,
        samples_per_open=4,
        raster_backend="mps",
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
        raster_backend="mps",
        raster_tile_size=20,
        raster_chunk_size=72,
    )

    assert scene.raster_backend == "mps"
    assert scene.raster_tile_size == 20
    assert scene.raster_chunk_size == 72
