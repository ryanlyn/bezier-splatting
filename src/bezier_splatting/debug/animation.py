"""GIF animation export for Bézier Splatting training runs.

Captures frames during optimization and exports an animated GIF with optional
multi-panel layouts (rendered image, error heatmap, loss chart). Uses PIL for
GIF encoding — no external video dependencies (no imageio, no ffmpeg).
"""

import json
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
        max_size_mb: Maximum GIF size in megabytes. Export adaptively reduces
            color palette, frame count, and panel size to stay under this cap.
    """

    layout: str = "standard"
    target_frames: int = 120
    fps: int = 10
    last_frame_hold: float = 3.0
    panel_size: int | None = None
    max_size_mb: float = 8.0


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
    svg_overlay: np.ndarray | None = None  # (H, W, 3) uint8 SVG curve render


class FrameRecorder:
    """Thread-safe frame accumulator for training animations.

    Call ``maybe_capture`` from the training loop. After training, call
    ``export`` to write an animated GIF.

    The recorder captures every frame it receives and downsamples to
    ``target_frames`` at export time. This is necessary because
    ``maybe_capture`` is typically called at sparse intervals (e.g. every
    ``display_every`` steps from the inspector), so a pre-computed capture
    schedule would miss most calls.
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
        self._pending_event: str | None = None

    def maybe_capture(
        self,
        step: int,
        total_steps: int,
        rendered: Tensor,
        loss: float,
        psnr: float,
        n_open: int,
        n_closed: int,
        svg_overlay: np.ndarray | None = None,
    ) -> None:
        """Capture a frame unconditionally.

        Thread-safe — safe to call from training threads. The rendered tensor
        is detached and moved to CPU immediately. Downsampling to
        ``target_frames`` happens at export time.

        Args:
            svg_overlay: Optional (H, W, 3) uint8 numpy array of the SVG
                curve overlay render. Included in standard/full layouts.
        """
        frame = FrameData(
            step=step,
            rendered=rendered.detach().cpu().float(),
            loss=loss,
            psnr=psnr,
            n_open=n_open,
            n_closed=n_closed,
            event=self._pending_event,
            svg_overlay=svg_overlay.copy() if svg_overlay is not None else None,
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

    def save(self, path: Path) -> Path:
        """Persist captured frames to a ``.pt`` file for deferred GIF export.

        The file contains all frame data (rendered tensors, SVG overlays,
        metrics) plus the target image and resolution — everything needed
        to reconstruct the GIF later via ``FrameRecorder.load()``.

        Returns the path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialised_frames = []
        for f in self._frames:
            serialised_frames.append({
                "step": f.step,
                "rendered": f.rendered,
                "loss": f.loss,
                "psnr": f.psnr,
                "n_open": f.n_open,
                "n_closed": f.n_closed,
                "event": f.event,
                "svg_overlay": torch.from_numpy(f.svg_overlay) if f.svg_overlay is not None else None,
            })

        torch.save({
            "target_np": torch.from_numpy(self._target_np),
            "H": self._H,
            "W": self._W,
            "frames": serialised_frames,
        }, path)
        return path

    @classmethod
    def load(cls, path: Path, config: AnimationConfig | None = None) -> "FrameRecorder":
        """Load a previously saved ``FrameRecorder`` from a ``.pt`` file.

        The returned recorder is ready for ``export()`` — call with any
        ``AnimationConfig`` to re-compose the GIF with different layout/fps.
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        config = config or AnimationConfig()
        rec = cls.__new__(cls)
        rec._config = config
        rec._target_np = data["target_np"].numpy()
        rec._H = data["H"]
        rec._W = data["W"]
        rec._lock = threading.Lock()
        rec._pending_event = None

        rec._frames = []
        for fd in data["frames"]:
            svg = fd["svg_overlay"].numpy() if fd["svg_overlay"] is not None else None
            rec._frames.append(FrameData(
                step=fd["step"],
                rendered=fd["rendered"],
                loss=fd["loss"],
                psnr=fd["psnr"],
                n_open=fd["n_open"],
                n_closed=fd["n_closed"],
                event=fd["event"],
                svg_overlay=svg,
            ))
        return rec

    def export(self, output_path: Path) -> Path:
        """Compose all frames and write an animated GIF + sidecar JSON.

        Returns the path to the written GIF file.
        """
        output_path = Path(output_path)
        if not self._frames:
            raise ValueError("No frames captured — nothing to export")

        # Downsample to target_frames if we captured more than needed.
        # Always keep first, last, and any topology event frames.
        frames = _downsample_frames(self._frames, self._config.target_frames)

        panel_size = self._config.panel_size
        if panel_size is None:
            # Default: scale up small training resolutions so GIF is readable.
            # Target ~256px panels; leave high-res training as-is.
            panel_size = max(256, self._H, self._W)
        panel_size = max(panel_size, 128)

        max_bytes = int(self._config.max_size_mb * 1024 * 1024)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected_frames = list(frames)
        selected_panel_size = panel_size
        selected_colors = 256

        for frame_ratio, panel_ratio, colors in _compression_plan():
            target_count = max(1, int(round(len(frames) * frame_ratio)))
            candidate_frames = _downsample_frames(frames, target_count)

            if panel_size is None:
                candidate_panel_size = None
            else:
                candidate_panel_size = max(64, int(round(panel_size * panel_ratio)))

            candidate_images, candidate_json = _compose_export_frames(
                candidate_frames,
                self._target_np,
                self._config.layout,
                candidate_panel_size,
            )
            candidate_durations = _build_durations(
                len(candidate_images),
                self._config.fps,
                self._config.last_frame_hold,
            )
            encoded = [_quantize_gif_frame(img, colors) for img in candidate_images]
            _save_gif(output_path, encoded, candidate_durations)

            if output_path.stat().st_size <= max_bytes:
                selected_frames = candidate_frames
                selected_panel_size = candidate_panel_size if candidate_panel_size is not None else panel_size
                selected_colors = colors
                json_frames = candidate_json
                break
        else:
            # Hard fallback: keep only the final frame at tiny size.
            fallback_frames = [frames[-1]]
            fallback_panel = 32 if panel_size is None else max(32, panel_size // 8)
            fallback_images, fallback_json = _compose_export_frames(
                fallback_frames,
                self._target_np,
                self._config.layout,
                fallback_panel,
            )
            fallback_durations = _build_durations(
                len(fallback_images),
                self._config.fps,
                self._config.last_frame_hold,
            )
            encoded = [_quantize_gif_frame(img, 16) for img in fallback_images]
            _save_gif(output_path, encoded, fallback_durations)
            selected_frames = fallback_frames
            selected_panel_size = fallback_panel
            selected_colors = 16
            json_frames = fallback_json

        # Sidecar JSON
        sidecar = {
            "target": None,
            "resolution": [self._H, self._W],
            "total_steps": selected_frames[-1].step if selected_frames else 0,
            "fps": self._config.fps,
            "layout": self._config.layout,
            "max_size_mb": self._config.max_size_mb,
            "final_size_bytes": output_path.stat().st_size,
            "palette_colors": selected_colors,
            "panel_size": selected_panel_size,
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


def _compression_plan() -> list[tuple[float, float, int]]:
    """Progressive (frame ratio, panel ratio, palette colors) compression plan."""
    return [
        (1.00, 1.00, 256),
        (1.00, 1.00, 192),
        (1.00, 1.00, 128),
        (0.85, 1.00, 128),
        (0.70, 1.00, 96),
        (0.55, 0.90, 96),
        (0.45, 0.80, 64),
        (0.35, 0.70, 64),
        (0.28, 0.60, 48),
        (0.22, 0.50, 32),
        (0.16, 0.40, 24),
        (0.12, 0.35, 16),
        (0.08, 0.30, 16),
        (0.05, 0.25, 16),
        (0.03, 0.20, 16),
        (0.02, 0.16, 16),
        (0.01, 0.12, 16),
    ]


def _compose_export_frames(
    frames: list[FrameData],
    target_np: np.ndarray,
    layout: str,
    panel_size: int | None,
) -> tuple[list[Image.Image], list[dict]]:
    """Compose frame images and JSON frame metadata for export."""
    pil_frames: list[Image.Image] = []
    json_frames: list[dict] = []
    all_losses = [f.loss for f in frames]
    all_psnrs = [f.psnr for f in frames]
    total_steps = frames[-1].step if frames else 0

    for i, frame in enumerate(frames):
        rendered_np = _tensor_to_uint8(frame.rendered)
        composed = _compose_frame(
            layout=layout,
            target_np=target_np,
            rendered_np=rendered_np,
            frame=frame,
            total_steps=total_steps,
            losses=all_losses[: i + 1],
            psnrs=all_psnrs[: i + 1],
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
    return pil_frames, json_frames


def _build_durations(frame_count: int, fps: int, last_frame_hold: float) -> list[int]:
    """Build per-frame GIF durations (ms), with an extended final frame."""
    base_duration = max(1, 1000 // max(1, fps))
    durations = [base_duration] * max(1, frame_count)
    hold_ms = int(max(1, round(last_frame_hold * 1000.0)))
    durations[-1] = hold_ms
    return durations


def _quantize_gif_frame(image: Image.Image, colors: int) -> Image.Image:
    """Quantize an RGB frame to a bounded palette for GIF compression."""
    colors = max(2, min(256, int(colors)))
    dither_none = getattr(Image, "Dither", None)
    dither_value = dither_none.NONE if dither_none is not None else Image.NONE
    return image.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors, dither=dither_value)


def _save_gif(path: Path, frames: list[Image.Image], durations: list[int]) -> None:
    """Write GIF to disk from quantized P-mode frames."""
    if not frames:
        raise ValueError("Cannot save empty GIF frame list")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _downsample_frames(frames: list[FrameData], target: int) -> list[FrameData]:
    """Downsample frames to at most *target*, keeping first, last, and events.

    Uses **logarithmic** spacing so the early (fast-changing) phase of
    optimisation gets many more frames than the later (converged) phase.
    If we have fewer than *target* frames, returns all of them.
    """
    if len(frames) <= target:
        return list(frames)

    n = len(frames)

    # Always keep first, last, and topology-event frames
    keep_indices: set[int] = {0, n - 1}
    for i, f in enumerate(frames):
        if f.event is not None:
            keep_indices.add(i)

    # Fill remaining budget with log-spaced indices.
    # Exponential mapping gives dense coverage at the start (where changes
    # are most dramatic) and sparse coverage toward convergence.
    budget = target - len(keep_indices)
    if budget > 0:
        candidates = [i for i in range(n) if i not in keep_indices]
        if candidates and budget < len(candidates):
            import math
            nc = len(candidates)
            log_max = math.log1p(nc)
            # Generate all unique candidate indices from log-spaced probes.
            # Use nc*4 probes to avoid gaps from int() collisions.
            probes = max(nc * 4, budget * 8)
            selected_set: set[int] = set()
            selected_order: list[int] = []
            for j in range(probes):
                t = j / max(probes - 1, 1)
                idx = int(math.expm1(t * log_max))
                idx = min(idx, nc - 1)
                if idx not in selected_set:
                    selected_set.add(idx)
                    selected_order.append(candidates[idx])
                    if len(selected_order) >= budget:
                        break
            keep_indices.update(selected_order)
        else:
            keep_indices.update(candidates)

    return [frames[i] for i in sorted(keep_indices)]


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


def _get_panels(
    target_np: np.ndarray,
    rendered_np: np.ndarray,
    frame: FrameData,
    panel_size: int | None,
) -> list[np.ndarray]:
    """Build the list of image panels for standard/full layouts.

    Returns [target, rendered, error] or [target, rendered, error, svg] if
    an SVG overlay is available. All panels are guaranteed to be the same size.
    """
    target_np = _resize_panel(target_np, panel_size)
    rendered_np = _resize_panel(rendered_np, panel_size)
    error_np = _compute_error_map(rendered_np, target_np)
    panels = [target_np, rendered_np, error_np]
    if frame.svg_overlay is not None:
        # Force SVG to match the other panels' dimensions (the SVG overlay
        # may arrive at a different resolution from _upscale_image).
        h, w = target_np.shape[:2]
        svg_pil = Image.fromarray(frame.svg_overlay).resize((w, h), Image.LANCZOS)
        panels.append(np.array(svg_pil))
    return panels


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
    """Multi-panel [target | rendered | error | svg?] with header bar."""
    panels = _get_panels(target_np, rendered_np, frame, panel_size)
    n_panels = len(panels)
    ph, pw = panels[0].shape[:2]
    gap = 2
    total_w = pw * n_panels + gap * (n_panels - 1)
    header_h = 28
    total_h = ph + header_h

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    header = _make_header_bar(total_w, frame, total_steps, bar_height=header_h)
    canvas.paste(header, (0, 0))

    for i, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (i * (pw + gap), header_h))

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
    panels = _get_panels(target_np, rendered_np, frame, panel_size)
    n_panels = len(panels)
    ph, pw = panels[0].shape[:2]
    gap = 2
    total_w = pw * n_panels + gap * (n_panels - 1)
    header_h = 28
    chart_h = max(ph // 2, 100)
    total_h = ph + header_h + chart_h

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    header = _make_header_bar(total_w, frame, total_steps, bar_height=header_h)
    canvas.paste(header, (0, 0))

    for i, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (i * (pw + gap), header_h))

    # Loss chart
    if len(losses) > 1:
        chart = _make_loss_chart(losses, psnrs, total_w, chart_h)
        canvas.paste(chart, (0, header_h + ph))

    return canvas
