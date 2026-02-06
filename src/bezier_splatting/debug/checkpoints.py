"""Checkpoint save / load / list for VectorGraphicsScene."""

import re
from pathlib import Path

import torch


def save_checkpoint(
    scene,
    step: int,
    metrics: dict,
    output_dir: Path,
) -> Path:
    """Save scene.state_dict() plus metadata to a checkpoint file.

    File is written to ``output_dir/checkpoints/step_{step:06d}.pt``.

    Returns:
        Path to the saved checkpoint file.
    """
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"step_{step:06d}.pt"
    payload = {
        "state_dict": scene.state_dict(),
        "step": step,
        "metrics": metrics,
        "n_open": scene.n_open,
        "n_closed": scene.n_closed,
        "num_cp_closed": int(scene.closed_interior_cp.shape[2] + 2),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[dict, int, dict]:
    """Load a checkpoint from disk.

    Args:
        path: Path to a ``.pt`` checkpoint file.
        device: Device to map tensors onto.

    Returns:
        (state_dict, step, metrics) tuple.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    return payload["state_dict"], payload["step"], payload["metrics"]


_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def list_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """List all checkpoints under ``output_dir/checkpoints/``, sorted by step.

    Returns:
        List of ``(step, path)`` tuples in ascending step order.
    """
    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return []

    results: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        m = _STEP_RE.search(p.name)
        if m:
            results.append((int(m.group(1)), p))

    results.sort(key=lambda x: x[0])
    return results
