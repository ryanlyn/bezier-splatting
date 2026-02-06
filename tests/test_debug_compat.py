"""Regression tests for debug utilities and checkpoint compatibility."""

from pathlib import Path

import torch

from bezier_splatting.debug.checkpoints import save_checkpoint
from bezier_splatting.debug.collectors import snapshot_scene
from bezier_splatting.debug.samples import load_image
from bezier_splatting.debug.viz import _scene_from_checkpoint, _scene_from_snapshot
from bezier_splatting.model import VectorGraphicsScene


def test_scene_from_checkpoint_handles_variable_closed_cp():
    scene = VectorGraphicsScene(n_open=2, n_closed=1, closed_cp=7, H=32, W=32)
    ckpt = {
        "state_dict": scene.state_dict(),
        "n_open": scene.n_open,
        "n_closed": scene.n_closed,
    }

    restored = _scene_from_checkpoint(ckpt)

    assert restored.closed_interior_cp.shape == scene.closed_interior_cp.shape
    assert torch.allclose(restored.closed_shared_pts, scene.closed_shared_pts)


def test_scene_from_snapshot_handles_variable_closed_cp():
    scene = VectorGraphicsScene(n_open=1, n_closed=2, closed_cp=6, H=32, W=32)
    snap = snapshot_scene(scene)

    restored = _scene_from_snapshot(snap)

    assert restored.closed_interior_cp.shape == scene.closed_interior_cp.shape
    assert torch.allclose(restored.closed_shared_pts, scene.closed_shared_pts)


def test_load_image_handles_rgba(tmp_path: Path):
    from PIL import Image

    path = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 128)).save(path)

    image = load_image(path, H=4, W=4)

    assert image.shape == (3, 4, 4)
    assert image.dtype == torch.float32
    center = image[:, 2, 2]
    expected = torch.tensor([1.0, 0.5, 0.5], dtype=torch.float32)
    assert torch.allclose(center, expected, atol=0.02)


def test_save_checkpoint_records_closed_cp_metadata(tmp_path: Path):
    scene = VectorGraphicsScene(n_open=0, n_closed=1, closed_cp=8, H=32, W=32)

    ckpt_path = save_checkpoint(scene, step=3, metrics={"loss": 1.0}, output_dir=tmp_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert payload["num_cp_closed"] == 8
