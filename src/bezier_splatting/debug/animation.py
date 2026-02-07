"""GIF animation export for Bézier Splatting training runs.

Captures frames during optimization and exports an animated GIF with optional
multi-panel layouts (rendered image, error heatmap, loss chart). Uses PIL for
GIF encoding — no external video dependencies (no imageio, no ffmpeg).
"""

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor


@dataclass
class AnimationConfig:
    """Configuration for training animation export.

    Attributes:
        layout: Panel layout — "minimal" (rendered only + text overlay),
            "standard" (target | rendered | error heatmap + header bar),
            "full" (standard + loss chart row).
        target_frames: Maximum number of frames to capture. Frames are spaced
            logarithmically so early rapid changes get more coverage.
        fps: Playback speed in frames per second.
        last_frame_hold: Hold the final frame this many seconds longer.
        panel_size: Resize panels to this dimension. None uses the training
            resolution. Minimum 128.
    """

    layout: str = "standard"
    target_frames: int = 120
    fps: int = 10
    last_frame_hold: float = 3.0
    panel_size: int | None = None


@dataclass
class FrameData:
    """Captured data for a single animation frame."""

    step: int
    rendered: Tensor  # (3, H, W) CPU float
    loss: float
    psnr: float
    n_open: int
    n_closed: int
    event: str | None = None  # "prune" or "densify"


class FrameRecorder:
    """Thread-safe frame accumulator for training animations.

    Call ``maybe_capture`` from the training loop. After training, call
    ``export`` to write an animated GIF.

    The recorder uses logarithmic frame scheduling: early training steps
    (where visual change is fastest) are sampled more densely than later
    steps. This gives a natural "fast start, slow finish" feel.
    """

    def __init__(
        self,
        config: AnimationConfig,
        target: Tensor,
        H: int,
        W: int,
    ):
        self._config = config
        self._target_np = _tensor_to_uint8(target)
        self._H = H
        self._W = W
        self._frames: list[FrameData] = []
        self._lock = threading.Lock()
        self._capture_steps: set[int] | None = None
        self._pending_event: str | None = None
        self._all_losses: list[float] = []
        self._all_psnrs: list[float] = []

    def maybe_capture(
        self,
        step: int,
        total_steps: int,
        rendered: Tensor,
        loss: float,
        psnr: float,
        n_open: int,
        n_closed: int,
    ) -> None:
        """Capture a frame if this step is in the schedule.

        Thread-safe — safe to call from training threads. The rendered tensor
        is detached and moved to CPU immediately.
        """
        self._all_losses.append(loss)
        self._all_psnrs.append(psnr)

        if self._capture_steps is None:
            self._capture_steps = _build_capture_schedule(
                total_steps, self._config.target_frames
            )

        if step not in self._capture_steps:
            return

        frame = FrameData(
            step=step,
            rendered=rendered.detach().cpu().float(),
            loss=loss,
            psnr=psnr,
            n_open=n_open,
            n_closed=n_closed,
            event=self._pending_event,
        )
        with self._lock:
            self._frames.append(frame)
            self._pending_event = None

    def record_topology_event(self, step: int, event_type: str) -> None:
        """Mark the next captured frame with a topology event label."""
        with self._lock:
            self._pending_event = event_type

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def export(self, output_path: Path) -> Path:
        """Compose all frames and write an animated GIF + sidecar JSON.

        Returns the path to the written GIF file.
        """
        output_path = Path(output_path)
        if not self._frames:
            raise ValueError("No frames captured — nothing to export")

        panel_size = self._config.panel_size
        if panel_size is not None:
            panel_size = max(panel_size, 128)

        pil_frames: list[Image.Image] = []
        json_frames: list[dict] = []

        for i, frame in enumerate(self._frames):
            rendered_np = _tensor_to_uint8(frame.rendered)
            losses_so_far = self._all_losses[: frame.step + 1]
            psnrs_so_far = self._all_psnrs[: frame.step + 1]

            composed = _compose_frame(
                layout=self._config.layout,
                target_np=self._target_np,
                rendered_np=rendered_np,
                frame=frame,
                total_steps=self._frames[-1].step,
                losses=losses_so_far,
                psnrs=psnrs_so_far,
                panel_size=panel_size,
            )
            pil_frames.append(composed)
            json_frames.append(
                {
                    "step": frame.step,
                    "loss": frame.loss,
                    "psnr": frame.psnr,
                    "n_open": frame.n_open,
                    "n_closed": frame.n_closed,
                    "event": frame.event,
                }
            )

        # Build per-frame durations (milliseconds)
        base_duration = 1000 // self._config.fps
        durations = [base_duration] * len(pil_frames)
        # Hold last frame longer
        durations[-1] = int(base_duration * self._config.last_frame_hold / (1.0 / self._config.fps))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
        )

        # Sidecar JSON
        sidecar = {
            "target": None,
            "resolution": [self._H, self._W],
            "total_steps": self._frames[-1].step if self._frames else 0,
            "fps": self._config.fps,
            "layout": self._config.layout,
            "frames": json_frames,
        }
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(sidecar, indent=2))

        return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tensor_to_uint8(tensor: Tensor) -> np.ndarray:
    """Convert a (3, H, W) or (H, W, 3) float tensor to (H, W, 3) uint8 numpy."""
    t = tensor.detach().cpu().float()
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return (t.clamp(0, 1) * 255).byte().numpy()


def _build_capture_schedule(total_steps: int, target_frames: int) -> set[int]:
    """Build a set of steps to capture, spaced logarithmically.

    Early steps are sampled more densely because visual change is fastest
    at the start of optimization. Always includes step 0 and the final step.
    """
    if target_frames <= 0:
        return set()
    if target_frames >= total_steps:
        return set(range(total_steps + 1))

    # Logarithmic spacing: more frames early, fewer late
    log_steps = np.logspace(0, np.log10(total_steps + 1), target_frames, dtype=float)
    steps = set(int(round(s - 1)) for s in log_steps)
    steps.add(0)
    steps.add(total_steps)
    # Clamp to valid range
    steps = {max(0, min(s, total_steps)) for s in steps}
    return steps


def _compute_error_map(rendered_np: np.ndarray, target_np: np.ndarray) -> np.ndarray:
    """Compute a colorized L1 error heatmap as (H, W, 3) uint8 numpy.

    Local implementation to avoid circular imports while the extraction
    from inspector.py to viz.py is in progress.
    """
    import matplotlib.cm as cm

    rendered_f = rendered_np.astype(np.float32) / 255.0
    target_f = target_np.astype(np.float32) / 255.0
    error = np.abs(rendered_f - target_f).mean(axis=2)  # (H, W)
    error_max = error.max()
    if error_max > 0:
        error_norm = error / error_max
    else:
        error_norm = error
    colored = (cm.hot(error_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def _get_font(size: int = 12) -> ImageFont.ImageFont:
    """Get a PIL font, falling back to the default bitmap font."""
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _draw_text_overlay(
    img: Image.Image,
    text: str,
    position: tuple[int, int] = (4, 4),
    font_size: int = 12,
) -> Image.Image:
    """Draw semi-transparent text overlay on a PIL image (returns a copy)."""
    img = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(font_size)

    # Measure text bounding box
    bbox = draw.textbbox(position, text, font=font)
    padding = 4
    bg_rect = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )
    draw.rectangle(bg_rect, fill=(0, 0, 0, 160))
    draw.text(position, text, fill=(255, 255, 255, 230), font=font)

    composited = Image.alpha_composite(img, overlay)
    return composited.convert("RGB")


def _make_header_bar(
    width: int,
    frame: FrameData,
    total_steps: int,
    bar_height: int = 28,
) -> Image.Image:
    """Create a dark header bar with step/loss/PSNR/curve count info."""
    bar = Image.new("RGB", (width, bar_height), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    font = _get_font(13)

    text = (
        f"Step {frame.step}/{total_steps}"
        f"    Loss: {frame.loss:.4f}"
        f"    PSNR: {frame.psnr:.1f} dB"
        f"    Curves: {frame.n_open}o + {frame.n_closed}c"
    )
    if frame.event:
        text += f"    [{frame.event.upper()}]"

    draw.text((6, 5), text, fill=(220, 220, 220), font=font)
    return bar


def _make_loss_chart(
    losses: list[float],
    psnrs: list[float],
    width: int,
    height: int,
) -> Image.Image:
    """Create a loss+PSNR dual chart as a PIL image.

    Uses matplotlib for rendering, then converts to PIL.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)
    steps = list(range(len(losses)))

    ax1.plot(steps, losses, "b-", linewidth=1)
    ax1.set_xlabel("Step", fontsize=8)
    ax1.set_ylabel("Loss", fontsize=8)
    if len(losses) > 1 and max(losses) > 0:
        ax1.set_yscale("log")
    ax1.set_title("Loss", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)

    ax2.plot(steps, psnrs, "g-", linewidth=1)
    ax2.set_xlabel("Step", fontsize=8)
    ax2.set_ylabel("PSNR (dB)", fontsize=8)
    ax2.set_title(f"PSNR: {psnrs[-1]:.1f} dB" if psnrs else "PSNR", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=7)

    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    return Image.fromarray(arr).resize((width, height), Image.LANCZOS)


def _resize_panel(img_np: np.ndarray, size: int | None) -> np.ndarray:
    """Resize a (H, W, 3) uint8 numpy image to (size, size) if size is given."""
    if size is None:
        return img_np
    pil = Image.fromarray(img_np)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil)


def _compose_frame(
    layout: str,
    target_np: np.ndarray,
    rendered_np: np.ndarray,
    frame: FrameData,
    total_steps: int,
    losses: list[float],
    psnrs: list[float],
    panel_size: int | None,
) -> Image.Image:
    """Compose a single animation frame according to the layout.

    Returns a PIL Image in RGB mode.
    """
    if layout == "minimal":
        return _compose_minimal(rendered_np, frame, total_steps, panel_size)
    elif layout == "standard":
        return _compose_standard(
            target_np, rendered_np, frame, total_steps, panel_size
        )
    elif layout == "full":
        return _compose_full(
            target_np, rendered_np, frame, total_steps, losses, psnrs, panel_size
        )
    else:
        raise ValueError(f"Unknown layout: {layout!r}. Use 'minimal', 'standard', or 'full'.")


def _compose_minimal(
    rendered_np: np.ndarray,
    frame: FrameData,
    total_steps: int,
    panel_size: int | None,
) -> Image.Image:
    """Rendered image with small text overlay (step + PSNR)."""
    rendered_np = _resize_panel(rendered_np, panel_size)
    img = Image.fromarray(rendered_np)
    h = img.height
    text = f"Step {frame.step}  PSNR {frame.psnr:.1f}"
    # Position at bottom-left
    return _draw_text_overlay(img, text, position=(4, h - 20), font_size=11)


def _compose_standard(
    target_np: np.ndarray,
    rendered_np: np.ndarray,
    frame: FrameData,
    total_steps: int,
    panel_size: int | None,
) -> Image.Image:
    """3-panel [target | rendered | error] with header bar."""
    target_np = _resize_panel(target_np, panel_size)
    rendered_np = _resize_panel(rendered_np, panel_size)
    error_np = _compute_error_map(rendered_np, target_np)

    ph, pw = rendered_np.shape[:2]
    gap = 2
    total_w = pw * 3 + gap * 2
    header_h = 28
    total_h = ph + header_h

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    # Header
    header = _make_header_bar(total_w, frame, total_steps, bar_height=header_h)
    canvas.paste(header, (0, 0))

    # Panels
    canvas.paste(Image.fromarray(target_np), (0, header_h))
    canvas.paste(Image.fromarray(rendered_np), (pw + gap, header_h))
    canvas.paste(Image.fromarray(error_np), (2 * (pw + gap), header_h))

    return canvas


def _compose_full(
    target_np: np.ndarray,
    rendered_np: np.ndarray,
    frame: FrameData,
    total_steps: int,
    losses: list[float],
    psnrs: list[float],
    panel_size: int | None,
) -> Image.Image:
    """Standard layout + loss chart row at the bottom."""
    target_np = _resize_panel(target_np, panel_size)
    rendered_np = _resize_panel(rendered_np, panel_size)
    error_np = _compute_error_map(rendered_np, target_np)

    ph, pw = rendered_np.shape[:2]
    gap = 2
    total_w = pw * 3 + gap * 2
    header_h = 28
    chart_h = max(ph // 2, 100)
    total_h = ph + header_h + chart_h

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    # Header
    header = _make_header_bar(total_w, frame, total_steps, bar_height=header_h)
    canvas.paste(header, (0, 0))

    # Image panels
    canvas.paste(Image.fromarray(target_np), (0, header_h))
    canvas.paste(Image.fromarray(rendered_np), (pw + gap, header_h))
    canvas.paste(Image.fromarray(error_np), (2 * (pw + gap), header_h))

    # Loss chart
    if len(losses) > 1:
        chart = _make_loss_chart(losses, psnrs, total_w, chart_h)
        canvas.paste(chart, (0, header_h + ph))

    return canvas
