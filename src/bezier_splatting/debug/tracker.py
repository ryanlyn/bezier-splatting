"""DebugTracker — unified logging for scalars, images, and snapshots.

Wraps trackio for scalar metrics. Falls back gracefully when trackio
is not installed (images and snapshots still save to disk).
"""

from pathlib import Path

import torch
from torch import Tensor

try:
    import trackio

    _HAS_TRACKIO = True
except ImportError:
    _HAS_TRACKIO = False


class DebugTracker:
    """Log scalars via trackio, save images and tensor snapshots to disk."""

    def __init__(
        self,
        project: str = "bezier-splatting",
        run_name: str | None = None,
        output_dir: str | Path = "debug_output",
        config: dict | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "snapshots").mkdir(exist_ok=True)

        self._trackio_active = False
        if _HAS_TRACKIO:
            trackio.init(project=project, name=run_name, config=config or {})
            self._trackio_active = True

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        """Log scalar metrics to trackio (no-op if trackio unavailable)."""
        if self._trackio_active:
            trackio.log(metrics, step=step)

    def log_image(self, step: int, name: str, figure_or_tensor) -> None:
        """Save a matplotlib Figure or image tensor as PNG.

        Args:
            step: Current optimization step.
            name: Descriptive name (used in filename).
            figure_or_tensor: Either a ``matplotlib.figure.Figure`` or a
                ``(C, H, W)`` / ``(H, W, C)`` / ``(H, W)`` image tensor.
        """
        path = self.output_dir / "images" / f"{step:06d}_{name}.png"

        # matplotlib Figure
        try:
            import matplotlib.figure

            if isinstance(figure_or_tensor, matplotlib.figure.Figure):
                figure_or_tensor.savefig(path, bbox_inches="tight", dpi=150)
                return
        except ImportError:
            pass

        # Tensor path
        if isinstance(figure_or_tensor, Tensor):
            _save_tensor_as_png(figure_or_tensor, path)

    def log_snapshot(self, step: int, name: str, data: dict) -> None:
        """Save arbitrary tensor data to a .pt file."""
        path = self.output_dir / "snapshots" / f"{step:06d}_{name}.pt"
        torch.save(data, path)

    def finish(self) -> Path:
        """Finalize tracking session and return the output directory."""
        if self._trackio_active:
            trackio.finish()
        return self.output_dir


def _save_tensor_as_png(tensor: Tensor, path: Path) -> None:
    """Save an image tensor as PNG via PIL.

    Accepts ``(C, H, W)``, ``(H, W, C)``, or ``(H, W)`` layouts.
    Values in [0, 1] are scaled to [0, 255].
    """
    from PIL import Image
    import numpy as np

    t = tensor.detach().cpu().float()

    # Normalize to (H, W, C) uint8
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)
    if t.ndim == 2:
        t = t.unsqueeze(-1)

    if t.max() <= 1.0:
        t = t * 255.0
    arr = t.clamp(0, 255).byte().numpy()

    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)

    Image.fromarray(arr).save(path)
