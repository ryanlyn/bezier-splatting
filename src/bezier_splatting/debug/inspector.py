"""Standalone Gradio inspector app for Bezier Splatting debug output.

Provides an interactive dashboard for running optimization and exploring results:
  - **Train** tab: select targets, configure params, run optimization
  - **Training Overview** tab: loss/PSNR curves, curve counts, filmstrip
  - **Checkpoint Explorer** tab: step slider, rendered images, ellipses, gradients
  - **Pixel Inspector** tab: click-to-inspect per-pixel Gaussian contributions
  - **Pruning Events** tab: before/after pruning diffs

Two launch modes:
  - ``launch_inspector()`` — fresh start, Train tab is the entry point
  - ``launch_inspector(debug_output_dir)`` — load existing results

Requires ``gradio>=5.0.0`` (listed in the ``debug`` optional dependency group).
"""

import math
import queue
import re
import tempfile
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

try:
    from gradio import SelectData as _GrSelectData
except ImportError:
    _GrSelectData = None

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..coords import model_to_pixel
from ..model import VectorGraphicsScene
from ..rasterizer import _build_covariance, _invert_2x2, rasterize
from ..sampling import GaussianParams
from .checkpoints import list_checkpoints
from .samples import (
    SUGGESTED_PARAMS,
    get_kodak_samples,
    get_sample_targets,
    load_image,
)
from .viz import (
    _scene_from_checkpoint,
    _tensor_to_numpy_image,
    render_ellipse_overlay,
    render_gradient_heatmap,
    render_layer_decomposition,
    render_prune_diff,
    render_training_filmstrip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_checkpoint_payload(path: Path) -> dict:
    """Load a raw checkpoint payload dict from disk."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_snapshot(path: Path) -> dict:
    """Load a snapshot .pt file from disk."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _list_snapshots(output_dir: Path, pattern: str = "*") -> list[tuple[int, str, Path]]:
    """List snapshots matching a name pattern, sorted by step.

    Returns:
        List of (step, name, path) tuples.
    """
    snap_dir = Path(output_dir) / "snapshots"
    if not snap_dir.exists():
        return []

    step_re = re.compile(r"^(\d+)_(.+)\.pt$")
    results: list[tuple[int, str, Path]] = []
    for p in snap_dir.iterdir():
        m = step_re.match(p.name)
        if m:
            name = m.group(2)
            if pattern == "*" or pattern in name:
                results.append((int(m.group(1)), name, p))
    results.sort(key=lambda x: x[0])
    return results


def _find_snapshot_near_step(
    snapshots: list[tuple[int, str, Path]],
    step: int,
) -> tuple[int, str, Path] | None:
    """Find the snapshot closest to the given step."""
    if not snapshots:
        return None
    return min(snapshots, key=lambda x: abs(x[0] - step))


def _render_curve_overlay(scene: VectorGraphicsScene, H: int, W: int) -> np.ndarray:
    """Render the true Bezier curves from a scene as an (H, W, 3) uint8 numpy image.

    Uses matplotlib Path/PathPatch to draw the exact cubic Bezier geometry
    (open curves as stroked paths, closed curves as filled regions).
    Curves are depth-sorted by area (largest first = background) to match
    the rasterizer's compositing order.

    Linewidths are converted from data-space pixels to matplotlib display
    points so that stroke widths visually match the Gaussian rendering.
    Closed curve fills include a border stroke sized to the boundary
    Gaussian sigma_y extent.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    from ..area import closed_curve_enclosed_area
    from ..bezier import evaluate_bezier

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=96)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # flip y so (0,0) is top-left
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Convert data-space pixels to matplotlib linewidth points.
    # With axis("off") + tight_layout(pad=0), axes span ~full figure width.
    # 1 data pixel = (fig_width_inches / W) inches = (fig_width_inches * 72 / W) points
    pts_per_px = fig.get_figwidth() * 72 / max(W, 1)

    # Collect (area, patch) pairs for depth sorting
    elements: list[tuple[float, PathPatch]] = []

    # -- Closed curves: filled regions (top boundary forward, bottom backward) --
    if scene.n_closed > 0:
        closed_opacities = torch.sigmoid(scene.closed_opacities).detach().cpu()

        # Compute first-intermediate-curve weight for boundary σ_y estimation
        rho = scene.closed_sampler.rho
        R_total = scene.closed_sampler.num_intermediate + 2
        bias = scene.closed_sampler.boundary_bias
        u1 = 1.0 / (R_total - 1)
        w1 = 0.5 - 0.5 * math.cos(math.pi * u1 ** (1.0 / bias))
        t_probe = torch.linspace(0, 1, 8)

        for i in range(scene.n_closed):
            opacity = closed_opacities[i].item()
            if opacity < 0.01:
                continue
            bcp_model = scene.closed_boundary_cp[i].detach().cpu()  # (2, CP, 2) in [-1, 1]
            bcp = model_to_pixel(bcp_model, H, W)  # (2, CP, 2) in pixel coords
            top = bcp[0]  # (CP, 2)
            bot = bcp[1]
            color = torch.sigmoid(scene.closed_colors[i]).detach().cpu().tolist()
            num_cp = top.shape[0]

            area = closed_curve_enclosed_area(bcp.unsqueeze(0))[0].item()

            # Estimate boundary σ_y: distance from boundary curve to
            # first intermediate curve, divided by ρ.  This determines
            # how far the rendered Gaussians bleed beyond the exact boundary.
            interp1_cp = (1 - w1) * top + w1 * bot  # (CP, 2)
            top_pts = evaluate_bezier(top.unsqueeze(0), t_probe)[0]
            int1_pts = evaluate_bezier(interp1_cp.unsqueeze(0), t_probe)[0]
            cross_d = torch.sqrt(((top_pts - int1_pts) ** 2).sum(-1) + 1e-12)
            sigma_y = (cross_d / rho).clamp(min=0.1).mean().item()
            # Stroke expands outward by ~2σ (half the linewidth goes outward)
            border_lw = 4 * sigma_y * pts_per_px

            verts = []
            codes = []

            # Top boundary forward
            verts.append(top[0].tolist())
            codes.append(MplPath.MOVETO)
            if num_cp == 4:
                for j in range(1, 4):
                    verts.append(top[j].tolist())
                    codes.append(MplPath.CURVE4)
            else:
                for j in range(1, num_cp):
                    verts.append(top[j].tolist())
                    codes.append(MplPath.LINETO)

            # Line to bottom end
            verts.append(bot[-1].tolist())
            codes.append(MplPath.LINETO)

            # Bottom boundary backward
            if num_cp == 4:
                for j in [2, 1, 0]:
                    verts.append(bot[j].tolist())
                    codes.append(MplPath.CURVE4)
            else:
                for j in range(num_cp - 2, -1, -1):
                    verts.append(bot[j].tolist())
                    codes.append(MplPath.LINETO)

            # CLOSEPOLY needs its own dummy vertex — don't overwrite the last CURVE4
            verts.append([0.0, 0.0])
            codes.append(MplPath.CLOSEPOLY)

            path = MplPath(verts, codes)
            patch = PathPatch(
                path, facecolor=color, edgecolor=color,
                linewidth=border_lw, alpha=opacity,
            )
            elements.append((area, patch))

    # -- Open curves: stroked paths (3 connected cubics) --
    if scene.n_open > 0:
        open_opacities = torch.sigmoid(scene.open_opacities).detach().cpu()
        mean_opacity = open_opacities.mean(dim=-1)
        for i in range(scene.n_open):
            opacity = mean_opacity[i].item()
            if opacity < 0.01:
                continue
            cp = model_to_pixel(scene.open_control_points[i].detach().cpu(), H, W)  # (10, 2)
            color = torch.sigmoid(scene.open_colors[i]).detach().cpu().tolist()
            sw = 0.5 + torch.sigmoid(scene.open_stroke_widths[i]).detach().cpu().item() * 4.5

            edge_len = torch.norm(cp[1:] - cp[:-1], dim=-1).sum().item()
            area = edge_len * sw

            verts = [cp[0].tolist()]
            codes = [MplPath.MOVETO]
            for seg in range(3):
                base = seg * 3
                for j in range(1, 4):
                    verts.append(cp[base + j].tolist())
                    codes.append(MplPath.CURVE4)

            path = MplPath(verts, codes)
            # Convert stroke width from data pixels to display points
            linewidth = sw * pts_per_px
            patch = PathPatch(
                path, facecolor="none", edgecolor=color,
                linewidth=linewidth, alpha=opacity, capstyle="round",
            )
            elements.append((area, patch))

    # Draw largest-area first (background), smallest on top (foreground)
    elements.sort(key=lambda x: x[0], reverse=True)
    for _, patch in elements:
        ax.add_patch(patch)

    fig.tight_layout(pad=0)
    arr = _fig_to_numpy(fig)  # closes fig
    return _upscale_image(arr, min_size=384)


def _fig_to_numpy(fig) -> np.ndarray:
    """Convert a matplotlib Figure to an (H, W, 3) uint8 numpy array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)
    plt.close(fig)
    return arr[:, :, :3]


def _render_scene_image(scene: VectorGraphicsScene, H: int, W: int) -> np.ndarray:
    """Render a scene to an (H, W, 3) numpy image."""
    with torch.no_grad():
        rendered = scene.forward(H, W)
    return (_tensor_to_numpy_image(rendered) * 255).astype(np.uint8)


def _tensor_to_display(t: Tensor, min_size: int = 0) -> np.ndarray:
    """Convert a (3, H, W) float tensor in [0, 1] to (H, W, 3) uint8 numpy for Gradio."""
    t_cpu = t.detach().cpu()
    img = (t_cpu.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    if min_size > 0:
        img = _upscale_image(img, min_size)
    return img


def _upscale_image(img: np.ndarray, min_size: int = 384) -> np.ndarray:
    """Upscale a small image so it displays well in Gradio panels."""
    from PIL import Image as PILImage

    h, w = img.shape[:2]
    if h >= min_size and w >= min_size:
        return img
    scale = max(min_size / h, min_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    pil = PILImage.fromarray(img)
    pil = pil.resize((new_w, new_h), PILImage.NEAREST)
    return np.array(pil)


def _compute_error_map(rendered: Tensor, target: Tensor) -> np.ndarray:
    """Compute a colorized L1 error heatmap as (H, W, 3) uint8 numpy."""
    import matplotlib.cm as cm

    error = (rendered.detach().cpu() - target.detach().cpu()).abs().mean(dim=0).numpy()  # (H, W)
    error_max = error.max()
    if error_max > 0:
        error_norm = error / error_max
    else:
        error_norm = error
    colored = (cm.hot(error_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def _make_loss_chart(
    losses: list[float],
    psnrs: list[float],
    current_step: int,
    total_steps: int,
) -> np.ndarray:
    """Create a matplotlib loss+PSNR dual chart, returned as (H, W, 3) uint8 numpy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
    steps = list(range(len(losses)))

    ax1.plot(steps, losses, "b-", linewidth=1)
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.set_title(f"Loss (step {current_step}/{total_steps})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, psnrs, "g-", linewidth=1)
    ax2.set_xlabel("Update")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title(f"PSNR: {psnrs[-1]:.1f} dB" if psnrs else "PSNR")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _fig_to_numpy(fig)


def _detach_gaussians(g: GaussianParams) -> GaussianParams:
    """Detach and move all GaussianParams tensors to CPU for safe cross-thread use."""
    return GaussianParams(
        means=g.means.detach().cpu(),
        scales=g.scales.detach().cpu(),
        rotations=g.rotations.detach().cpu(),
        colors=g.colors.detach().cpu(),
        opacities=g.opacities.detach().cpu(),
        curve_ids=g.curve_ids.detach().cpu(),
    )


def compute_pixel_contributions(
    gaussians: GaussianParams,
    px: float,
    py: float,
) -> list[dict]:
    """Find Gaussians contributing to pixel (px, py).

    For each Gaussian i:
    - d = [px - means[i,0], py - means[i,1]]
    - Build 2x2 covariance: sigma = R @ diag(sx^2, sy^2) @ R.T
    - sigma_inv = inverse(sigma)
    - mahal_sq = d @ sigma_inv @ d
    - alpha_i = sigmoid(opacities[i]) * exp(-0.5 * mahal_sq)
    - Keep if mahal_sq < 9 (within 3 sigma)

    Returns sorted by original index (compositing order).
    """
    G = gaussians.means.shape[0]
    if G == 0:
        return []

    means = gaussians.means.detach().float()
    scales = gaussians.scales.detach().float()
    rotations = gaussians.rotations.detach().float()
    opacities = gaussians.opacities.detach().float()
    colors = gaussians.colors.detach().float()
    curve_ids = gaussians.curve_ids.detach()

    cov = _build_covariance(scales, rotations)
    inv_cov, _det = _invert_2x2(cov)

    pixel = torch.tensor([px, py], dtype=torch.float32)
    d = pixel.unsqueeze(0) - means

    d_transformed = torch.einsum("gi,gij->gj", d, inv_cov)
    mahal_sq = (d * d_transformed).sum(dim=-1)

    within_mask = mahal_sq < 9.0
    if not within_mask.any():
        return []

    indices = torch.where(within_mask)[0]
    alpha_sig = torch.sigmoid(opacities[indices])
    alpha_vals = alpha_sig * torch.exp(-0.5 * mahal_sq[indices])

    results: list[dict] = []
    for i, idx in enumerate(indices.tolist()):
        results.append({
            "index": idx,
            "curve_id": int(curve_ids[idx].item()),
            "alpha": float(alpha_vals[i].item()),
            "color": colors[idx].tolist(),
            "sigma_x": float(scales[idx, 0].item()),
            "sigma_y": float(scales[idx, 1].item()),
            "rotation_deg": float(math.degrees(rotations[idx].item())),
            "mahal_sq": float(mahal_sq[idx].item()),
        })

    results.sort(key=lambda x: x["index"])
    return results


# ---------------------------------------------------------------------------
# Data scanning (shared by result tabs)
# ---------------------------------------------------------------------------


def _scan_output_dir(output_dir: Path | None) -> dict:
    """Scan a debug output directory for available data.

    Returns a dict with keys: ckpt_list, grad_snaps, prune_before_snaps,
    prune_after_snaps, steps, step_min, step_max.
    Returns empty/zero values if output_dir is None or doesn't exist.
    """
    if output_dir is None or not output_dir.exists():
        return {
            "ckpt_list": [],
            "grad_snaps": [],
            "prune_before_snaps": [],
            "prune_after_snaps": [],
            "steps": [],
            "step_min": 0,
            "step_max": 1,
        }

    ckpt_list = list_checkpoints(output_dir)
    grad_snaps = _list_snapshots(output_dir, "grad_stats")
    prune_before_snaps = _list_snapshots(output_dir, "prune_before")
    prune_after_snaps = _list_snapshots(output_dir, "prune_after")

    steps = [s for s, _p in ckpt_list]
    step_min = steps[0] if steps else 0
    step_max = steps[-1] if steps else 1

    return {
        "ckpt_list": ckpt_list,
        "grad_snaps": grad_snaps,
        "prune_before_snaps": prune_before_snaps,
        "prune_after_snaps": prune_after_snaps,
        "steps": steps,
        "step_min": step_min,
        "step_max": step_max,
    }


def _resolve_scan_context(app_state: dict, output_dir_str: str | None) -> tuple[Path | None, dict]:
    """Resolve output dir + scanned artifacts from either UI state or shared app state."""
    normalized = str(output_dir_str).strip() if output_dir_str is not None else ""
    if normalized and normalized.lower() != "none":
        output_dir = Path(output_dir_str)
        return output_dir, _scan_output_dir(output_dir)
    output_dir = app_state.get("output_dir")
    cached = app_state.get("scan")
    if cached is None:
        cached = _scan_output_dir(output_dir)
    return output_dir, cached


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------


def create_inspector_app(debug_output_dir: str | Path | None = None):
    """Create the Gradio inspector app.

    Args:
        debug_output_dir: Path to the debug output from a training run
                         (contains checkpoints/, snapshots/, images/ subdirs).
                         If None, start with just the Train tab.

    Returns:
        A ``gr.Blocks`` application instance.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is required for the inspector app. "
            "Install it with: uv pip install 'bezier-splatting[debug]'"
        )

    output_dir = Path(debug_output_dir) if debug_output_dir is not None else None

    # Mutable state shared across tabs, wrapped in a dict so closures can mutate
    app_state = {
        "output_dir": output_dir,
        "scan": _scan_output_dir(output_dir),
    }

    # Default resolution
    default_H, default_W = 64, 64

    with gr.Blocks(title="Bezier Splatting Inspector") as app:
        gr.Markdown("# Bezier Splatting Debug Inspector")

        if output_dir is not None:
            gr.Markdown(f"**Loaded results from:** `{output_dir.resolve()}`")

        with gr.Row():
            render_h = gr.Number(value=default_H, label="Render H", precision=0)
            render_w = gr.Number(value=default_W, label="Render W", precision=0)

        # Shared state for output_dir (updated after training completes)
        output_dir_state = gr.State(value=str(output_dir) if output_dir else None)

        # ================================================================
        # Tab 0: Train
        # ================================================================
        with gr.Tab("Train"):
            _build_train_tab(gr, app, app_state, render_h, render_w, output_dir_state)

        # ================================================================
        # Tab 1: Training Overview
        # ================================================================
        with gr.Tab("Training Overview"):
            _build_training_overview_tab(gr, app_state, render_h, render_w, output_dir_state)

        # ================================================================
        # Tab 2: Checkpoint Explorer
        # ================================================================
        with gr.Tab("Checkpoint Explorer"):
            _build_checkpoint_explorer_tab(gr, app_state, render_h, render_w, output_dir_state)

        # ================================================================
        # Tab 3: Pixel Inspector
        # ================================================================
        with gr.Tab("Pixel Inspector"):
            _build_pixel_inspector_tab(gr, app_state, render_h, render_w, output_dir_state)

        # ================================================================
        # Tab 4: Pruning Events
        # ================================================================
        with gr.Tab("Pruning Events"):
            _build_pruning_tab(gr, app_state, render_h, render_w, output_dir_state)

    return app


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------


def _build_train_tab(gr, app, app_state, render_h, render_w, output_dir_state):
    """Tab 0: Image selection, parameters, live training visualization."""

    sample_targets = get_sample_targets()
    sample_names = list(sample_targets.keys())
    kodak_samples = get_kodak_samples()
    kodak_names = list(kodak_samples.keys())

    # -- Image Selection --
    gr.Markdown("## Image Selection")

    with gr.Row():
        with gr.Column(scale=2):
            source_radio = gr.Radio(
                choices=["Built-in Samples", "Kodak Samples", "Upload Image"],
                value="Built-in Samples",
                label="Image Source",
            )

            sample_dropdown = gr.Dropdown(
                choices=sample_names,
                value=sample_names[0] if sample_names else None,
                label="Built-in Target",
                visible=True,
            )

            kodak_dropdown = gr.Dropdown(
                choices=kodak_names if kodak_names else ["(no Kodak images found in samples/)"],
                value=kodak_names[0] if kodak_names else None,
                label="Kodak Image",
                visible=False,
                interactive=bool(kodak_names),
            )

            upload_image = gr.Image(
                label="Upload Image",
                type="filepath",
                visible=False,
            )

        with gr.Column(scale=1):
            preview_image = gr.Image(label="Target Preview", type="numpy", interactive=False)

    # -- Parameters --
    gr.Markdown("## Parameters")

    first_params = SUGGESTED_PARAMS.get(sample_names[0], {}) if sample_names else {}

    with gr.Row():
        resolution_slider = gr.Slider(
            minimum=64, maximum=512, value=64, step=64,
            label="Resolution (H = W)",
        )
        n_open_slider = gr.Slider(
            minimum=0, maximum=64, value=first_params.get("n_open", 16), step=1,
            label="Open Curves",
        )
        n_closed_slider = gr.Slider(
            minimum=0, maximum=64, value=first_params.get("n_closed", 8), step=1,
            label="Closed Curves",
        )
        steps_slider = gr.Slider(
            minimum=200, maximum=5000, value=first_params.get("steps", 1000), step=100,
            label="Training Steps",
        )

    with gr.Row():
        lr_scale_slider = gr.Slider(
            minimum=0.1, maximum=10.0, value=1.0, step=0.1,
            label="LR Scale (multiplier on all learning rates)",
        )

    # -- Run --
    gr.Markdown("## Run")

    with gr.Row():
        train_btn = gr.Button("Start Training", variant="primary", size="lg")
        stop_btn = gr.Button("Stop Training", variant="stop", size="lg")

    status_text = gr.Markdown("*Ready to train.*")

    # -- Live visualization panels --
    with gr.Row(equal_height=True):
        live_rendered = gr.Image(label="Rendered (live)", type="numpy", interactive=False, height=384)
        live_error = gr.Image(label="Error Heatmap (live)", type="numpy", interactive=False, height=384)
        live_ellipses = gr.Image(label="Gaussian Ellipses", type="numpy", interactive=False, height=384)
        live_svg = gr.Image(label="SVG Curves", type="numpy", interactive=False, height=384)

    live_chart = gr.Image(label="Loss / PSNR", type="numpy", interactive=False)

    log_output = gr.Textbox(label="Training Log", lines=8, max_lines=20, interactive=False)

    # -- Final result gallery (shown after training completes) --
    result_gallery = gr.Gallery(label="Training Result", columns=3, height="auto")

    # Shared mutable state for stop signaling across Gradio threads.
    # The generator (run_training) creates a fresh Event each run and stores it
    # here. The stop button handler reads and sets it.
    train_state = gr.State(value={"event": None})

    # -- Event handlers --

    def on_source_change(source):
        """Toggle visibility of source-specific controls."""
        return (
            gr.update(visible=(source == "Built-in Samples")),
            gr.update(visible=(source == "Kodak Samples")),
            gr.update(visible=(source == "Upload Image")),
        )

    source_radio.change(
        fn=on_source_change,
        inputs=[source_radio],
        outputs=[sample_dropdown, kodak_dropdown, upload_image],
    )

    def on_sample_change(sample_name, resolution):
        """Update preview and suggested params when a built-in sample is selected."""
        if sample_name not in sample_targets:
            return None, gr.update(), gr.update(), gr.update()

        res = int(resolution)
        target = sample_targets[sample_name](res, res)
        preview = _tensor_to_display(target)

        params = SUGGESTED_PARAMS.get(sample_name, {})
        return (
            preview,
            gr.update(value=params.get("n_open", 16)),
            gr.update(value=params.get("n_closed", 8)),
            gr.update(value=params.get("steps", 1000)),
        )

    sample_dropdown.change(
        fn=on_sample_change,
        inputs=[sample_dropdown, resolution_slider],
        outputs=[preview_image, n_open_slider, n_closed_slider, steps_slider],
    )

    def on_kodak_change(kodak_name, resolution):
        """Update preview when a Kodak image is selected."""
        if kodak_name not in kodak_samples:
            return None
        res = int(resolution)
        target = load_image(kodak_samples[kodak_name], res, res)
        return _tensor_to_display(target)

    kodak_dropdown.change(
        fn=on_kodak_change,
        inputs=[kodak_dropdown, resolution_slider],
        outputs=[preview_image],
    )

    def on_upload_change(filepath, resolution):
        """Update preview when an image is uploaded."""
        if filepath is None:
            return None
        res = int(resolution)
        target = load_image(filepath, res, res)
        return _tensor_to_display(target)

    upload_image.change(
        fn=on_upload_change,
        inputs=[upload_image, resolution_slider],
        outputs=[preview_image],
    )

    def on_resolution_change(resolution, source, sample_name, kodak_name, upload_path):
        """Re-generate preview at new resolution."""
        res = int(resolution)
        if source == "Built-in Samples" and sample_name in sample_targets:
            target = sample_targets[sample_name](res, res)
        elif source == "Kodak Samples" and kodak_name in kodak_samples:
            target = load_image(kodak_samples[kodak_name], res, res)
        elif source == "Upload Image" and upload_path is not None:
            target = load_image(upload_path, res, res)
        else:
            return None
        return _tensor_to_display(target)

    resolution_slider.change(
        fn=on_resolution_change,
        inputs=[resolution_slider, source_radio, sample_dropdown, kodak_dropdown, upload_image],
        outputs=[preview_image],
    )

    def run_training(
        state,
        source, sample_name, kodak_name, upload_path,
        resolution, n_open, n_closed, steps, lr_scale,
    ):
        """Generator that runs fit_image in a thread and yields live UI updates."""
        from ..metrics import compute_psnr
        from ..optimization import fit_image

        res = int(resolution)
        n_open = int(n_open)
        n_closed = int(n_closed)
        steps = int(steps)
        lr_scale = float(lr_scale)
        H = W = res
        current_output = str(app_state["output_dir"]) if app_state.get("output_dir") is not None else None

        # Resolve target image
        if source == "Built-in Samples":
            if sample_name not in sample_targets:
                yield (state, "*Error: no sample selected.*", None, None, None,
                       None, None, "", [], current_output)
                return
            target = sample_targets[sample_name](res, res)
        elif source == "Kodak Samples":
            if kodak_name not in kodak_samples:
                yield (state, "*Error: no Kodak image selected.*", None, None, None,
                       None, None, "", [], current_output)
                return
            target = load_image(kodak_samples[kodak_name], res, res)
        elif source == "Upload Image":
            if upload_path is None:
                yield (state, "*Error: no image uploaded.*", None, None, None,
                       None, None, "", [], current_output)
                return
            target = load_image(upload_path, res, res)
        else:
            yield (state, "*Error: unknown source.*", None, None, None,
                   None, None, "", [], current_output)
            return

        output_dir = Path(tempfile.mkdtemp(prefix="bezier_debug_"))

        stop_event = threading.Event()
        state["event"] = stop_event
        data_queue: queue.Queue[dict | None] = queue.Queue()
        result_holder: list[VectorGraphicsScene | None] = [None]

        display_every = max(1, steps // 40)
        ellipse_interval = display_every

        def callback(step: int, loss: float, scene: VectorGraphicsScene) -> bool | None:
            if stop_event.is_set():
                return False

            if step % display_every == 0 or step == steps - 1:
                with torch.no_grad():
                    rendered = scene(H, W)
                    gaussians = scene.get_gaussians(H, W)

                entry: dict = {
                    "step": step,
                    "loss": loss,
                    "rendered": rendered.detach().cpu(),
                    "n_open": scene.n_open,
                    "n_closed": scene.n_closed,
                    "gaussians": None,
                    "svg_image": None,
                }
                if gaussians is not None and step % ellipse_interval == 0:
                    entry["gaussians"] = _detach_gaussians(gaussians)

                if step % ellipse_interval == 0:
                    try:
                        entry["svg_image"] = _render_curve_overlay(scene, H, W)
                    except Exception:
                        pass

                data_queue.put(entry)
            return None

        error_holder: list[str | None] = [None]
        log_lines: list[str] = []

        class _LogCapture:
            """Tee stdout to both the original stream and a shared list.

            Delegates all unknown attributes to the original stream so that
            libraries like trackio/logging can configure formatters normally.
            """
            def __init__(self, original, lines):
                self.original = original
                self.lines = lines
            def write(self, text):
                self.original.write(text)
                if text.strip():
                    self.lines.append(text.rstrip())
                return len(text)
            def flush(self):
                self.original.flush()
            def __getattr__(self, name):
                return getattr(self.original, name)

        def train_thread():
            import sys
            import traceback
            capture = _LogCapture(sys.stdout, log_lines)
            sys.stdout = capture
            try:
                result_holder[0] = fit_image(
                    target,
                    n_open=n_open,
                    n_closed=n_closed,
                    steps=steps,
                    log_every=max(1, steps // 10),
                    lr_scale=lr_scale,
                    callback=callback,
                    debug=str(output_dir),
                )
            except Exception as e:
                tb = traceback.format_exc()
                capture.original.write(f"[bezier-debug] Training error:\n{tb}")
                msg = f"{type(e).__name__}: {e}"
                error_holder[0] = msg
                data_queue.put({"error": msg})
            finally:
                sys.stdout = capture.original
                data_queue.put(None)

        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()

        loss_history: list[float] = []
        psnr_history: list[float] = []
        last_ellipse_np: np.ndarray | None = None
        last_svg_np: np.ndarray | None = None
        last_rendered_np: np.ndarray | None = None
        last_error_np: np.ndarray | None = None
        last_chart_np: np.ndarray | None = None

        # Initial yield: show "training started" status
        yield (state, "**Training started...**", None, None, None,
               None, None, "", [], str(output_dir))

        while True:
            try:
                data = data_queue.get(timeout=1.0)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

            if data is None:
                break

            if "error" in data:
                yield (state, f"**Error:** {data['error']}", last_rendered_np,
                       last_error_np, last_ellipse_np, last_svg_np, last_chart_np,
                       "\n".join(log_lines), [], str(output_dir))
                break

            step = data["step"]
            loss = data["loss"]
            rendered_cpu = data["rendered"]

            rendered_np = _tensor_to_display(rendered_cpu, min_size=384)
            error_np = _upscale_image(_compute_error_map(rendered_cpu, target), min_size=384)
            last_rendered_np = rendered_np
            last_error_np = error_np

            loss_history.append(loss)
            psnr = compute_psnr(rendered_cpu, target).item()
            psnr_history.append(psnr)

            chart_np = _make_loss_chart(loss_history, psnr_history, step, steps)
            last_chart_np = chart_np

            gaussians = data.get("gaussians")
            if gaussians is not None:
                fig_ell = render_ellipse_overlay(
                    gaussians, H, W, background=rendered_cpu,
                )
                ellipse_np = _fig_to_numpy(fig_ell)
                last_ellipse_np = ellipse_np

            svg_image = data.get("svg_image")
            if svg_image is not None:
                last_svg_np = svg_image

            status = (
                f"**Step {step}/{steps}** | Loss: {loss:.5f} | "
                f"PSNR: {psnr:.1f} dB | "
                f"Curves: {data['n_open']} open + {data['n_closed']} closed"
            )

            yield (state, status, last_rendered_np, last_error_np,
                   last_ellipse_np, last_svg_np, last_chart_np,
                   "\n".join(log_lines), [], str(output_dir))

        thread.join(timeout=30)

        # Final yield with complete results
        scene = result_holder[0]
        if scene is not None:
            with torch.no_grad():
                final_rendered = scene(H, W)
                final_psnr = compute_psnr(final_rendered, target).item()

            final_rendered_np = _tensor_to_display(final_rendered, min_size=384)
            final_error_np = _upscale_image(_compute_error_map(final_rendered.cpu(), target), min_size=384)

            if loss_history:
                final_chart = _make_loss_chart(
                    loss_history, psnr_history, steps - 1, steps,
                )
            else:
                final_chart = last_chart_np

            # Build result gallery
            target_np = _tensor_to_display(target)
            gallery_images = [
                (target_np, "Target"),
                (final_rendered_np, "Rendered"),
                (final_error_np, "Error Heatmap"),
            ]

            # Update shared app state
            app_state["output_dir"] = output_dir
            app_state["scan"] = _scan_output_dir(output_dir)

            stopped = stop_event.is_set()
            last_step = loss_history[-1] if loss_history else 0
            status_prefix = "**Training stopped.**" if stopped else "**Training complete.**"
            final_status = (
                f"{status_prefix} PSNR: {final_psnr:.1f} dB | "
                f"Open: {scene.n_open}, Closed: {scene.n_closed} | "
                f"Results saved to `{output_dir}`"
            )

            final_log = "\n".join(log_lines)
            yield (state, final_status, final_rendered_np, final_error_np,
                   last_ellipse_np, last_svg_np, final_chart, final_log,
                   gallery_images, str(output_dir))
        else:
            err_msg = error_holder[0] or "Unknown error"
            yield (state, f"**Training failed:** `{err_msg}`",
                   last_rendered_np, last_error_np, last_ellipse_np,
                   last_svg_np, last_chart_np, "\n".join(log_lines), [], str(output_dir))

    def stop_training(state):
        """Signal the training thread to stop early."""
        event = state.get("event")
        if event is not None:
            event.set()
        return state, "**Stopping training...**"

    train_event = train_btn.click(
        fn=run_training,
        inputs=[
            train_state,
            source_radio, sample_dropdown, kodak_dropdown, upload_image,
            resolution_slider, n_open_slider, n_closed_slider, steps_slider, lr_scale_slider,
        ],
        outputs=[
            train_state, status_text, live_rendered, live_error,
            live_ellipses, live_svg, live_chart, log_output, result_gallery, output_dir_state,
        ],
    )

    stop_btn.click(
        fn=stop_training,
        inputs=[train_state],
        outputs=[train_state, status_text],
        cancels=[train_event],
    )

    # Generate initial preview on app load
    if sample_names:
        app.load(
            fn=lambda: on_sample_change(sample_names[0], 64),
            outputs=[preview_image, n_open_slider, n_closed_slider, steps_slider],
        )


def _build_training_overview_tab(gr, app_state, render_h, render_w, output_dir_state):
    """Tab 1: loss/PSNR curves, curve counts, filmstrip."""

    overview_gallery = gr.Gallery(label="Training Overview", columns=1, height="auto")
    btn = gr.Button("Generate Overview")

    def generate_overview(H, W, output_dir_str):
        H, W = int(H), int(W)

        # Re-scan in case training just completed
        output_dir, scan = _resolve_scan_context(app_state, output_dir_str)

        ckpt_list = scan["ckpt_list"]
        if not ckpt_list and output_dir is None:
            return [(np.zeros((100, 300, 3), dtype=np.uint8), "No results loaded. Run training first or launch with a debug directory.")]

        images = []

        if ckpt_list:
            steps_arr: list[int] = []
            losses: list[float] = []
            psnrs: list[float] = []
            n_opens: list[int] = []
            n_closeds: list[int] = []

            for step, path in ckpt_list:
                ckpt = _load_checkpoint_payload(path)
                steps_arr.append(step)
                metrics = ckpt.get("metrics", {})
                losses.append(metrics.get("loss", float("nan")))
                psnrs.append(metrics.get("psnr", float("nan")))
                n_opens.append(ckpt.get("n_open", 0))
                n_closeds.append(ckpt.get("n_closed", 0))

            # Loss + PSNR plot
            fig_metrics, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(steps_arr, losses, "b-", linewidth=1)
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training Loss")
            ax1.grid(True, alpha=0.3)

            valid_psnr = [(s, p) for s, p in zip(steps_arr, psnrs) if not math.isnan(p)]
            if valid_psnr:
                ax2.plot(
                    [s for s, _ in valid_psnr],
                    [p for _, p in valid_psnr],
                    "r-", linewidth=1,
                )
            ax2.set_xlabel("Step")
            ax2.set_ylabel("PSNR (dB)")
            ax2.set_title("PSNR")
            ax2.grid(True, alpha=0.3)

            fig_metrics.tight_layout()
            images.append(_fig_to_numpy(fig_metrics))

            # Curve count plot
            fig_counts, ax = plt.subplots(1, 1, figsize=(6, 3))
            ax.plot(steps_arr, n_opens, "g-", label="Open", linewidth=1)
            ax.plot(steps_arr, n_closeds, "m-", label="Closed", linewidth=1)
            total = [a + b for a, b in zip(n_opens, n_closeds)]
            ax.plot(steps_arr, total, "k--", label="Total", linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel("Count")
            ax.set_title("Curve Counts")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig_counts.tight_layout()
            images.append(_fig_to_numpy(fig_counts))

        # Filmstrip
        if output_dir is not None:
            fig_film = render_training_filmstrip(output_dir, H, W)
            images.append(_fig_to_numpy(fig_film))

        return images

    btn.click(
        fn=generate_overview,
        inputs=[render_h, render_w, output_dir_state],
        outputs=overview_gallery,
    )


def _build_checkpoint_explorer_tab(gr, app_state, render_h, render_w, output_dir_state):
    """Tab 2: step slider, rendered image, ellipse overlay, gradient heatmap."""

    gr.Markdown("*Load checkpoints to explore. Run training first or launch with a debug directory.*")

    step_input = gr.Number(value=0, label="Training Step", precision=0)
    load_btn = gr.Button("Load Step")

    with gr.Row():
        rendered_image = gr.Image(label="Rendered Image", type="numpy")
        ellipse_image = gr.Image(label="Gaussian Ellipses", type="numpy")

    grad_image = gr.Image(label="Gradient Heatmap", type="numpy")

    decompose_btn = gr.Button("Generate Layer Decomposition (slow)")
    decompose_gallery = gr.Gallery(label="Layer Decomposition", columns=1)

    # State to hold loaded checkpoint info
    loaded_state = gr.State(value=None)

    def on_load_step(step_val, H, W, output_dir_str):
        H, W = int(H), int(W)
        step_val = int(float(step_val))

        _output_dir, scan = _resolve_scan_context(app_state, output_dir_str)

        ckpt_list = scan["ckpt_list"]
        grad_snaps = scan["grad_snaps"]

        if not ckpt_list:
            return None, None, None, None

        steps = [s for s, _ in ckpt_list]
        step_to_path = {s: p for s, p in ckpt_list}

        closest_step = min(steps, key=lambda s: abs(s - step_val))
        path = step_to_path[closest_step]

        ckpt = _load_checkpoint_payload(path)
        scene = _scene_from_checkpoint(ckpt)

        rendered_np = _render_scene_image(scene, H, W)

        with torch.no_grad():
            gaussians = scene.get_gaussians(H, W)

        if gaussians is not None:
            rendered_tensor = torch.from_numpy(rendered_np).permute(2, 0, 1).float() / 255.0
            fig_ell = render_ellipse_overlay(gaussians, H, W, background=rendered_tensor)
            ellipse_np = _fig_to_numpy(fig_ell)
        else:
            ellipse_np = rendered_np

        grad_np = None
        if grad_snaps:
            nearest = _find_snapshot_near_step(grad_snaps, closest_step)
            if nearest is not None:
                grad_data = _load_snapshot(nearest[2])
                rendered_tensor = torch.from_numpy(rendered_np).permute(2, 0, 1).float() / 255.0
                fig_grad = render_gradient_heatmap(scene, grad_data, H, W, background=rendered_tensor)
                grad_np = _fig_to_numpy(fig_grad)

        state_info = {"closest_step": closest_step, "ckpt": ckpt}
        return rendered_np, ellipse_np, grad_np, state_info

    load_btn.click(
        fn=on_load_step,
        inputs=[step_input, render_h, render_w, output_dir_state],
        outputs=[rendered_image, ellipse_image, grad_image, loaded_state],
    )

    def on_decompose(state_info, H, W):
        H, W = int(H), int(W)
        if state_info is None:
            return []

        ckpt = state_info["ckpt"]
        scene = _scene_from_checkpoint(ckpt)
        fig = render_layer_decomposition(scene, H, W)
        return [_fig_to_numpy(fig)]

    decompose_btn.click(
        fn=on_decompose,
        inputs=[loaded_state, render_h, render_w],
        outputs=decompose_gallery,
    )


def _build_pixel_inspector_tab(gr, app_state, render_h, render_w, output_dir_state):
    """Tab 3: click on rendered image to see per-pixel Gaussian contributions."""

    gr.Markdown("*Load a checkpoint, then click on the image or enter coordinates to inspect.*")

    with gr.Row():
        step_input = gr.Number(value=0, label="Checkpoint Step", precision=0)
        load_btn = gr.Button("Load Checkpoint")

    scene_state = gr.State(value=None)

    clickable_image = gr.Image(label="Click on rendered image", type="numpy", interactive=False)

    with gr.Row():
        px_x = gr.Number(value=0, label="Pixel X", precision=1)
        px_y = gr.Number(value=0, label="Pixel Y", precision=1)
    inspect_btn = gr.Button("Inspect Pixel")

    contrib_table = gr.Dataframe(
        headers=[
            "Index", "Curve ID", "Alpha", "Color R", "Color G", "Color B",
            "Sigma X", "Sigma Y", "Rotation (deg)", "Mahal^2",
        ],
        label="Contributing Gaussians (front-to-back)",
    )
    overlay_image = gr.Image(label="Contributing Ellipses", type="numpy")

    def load_ckpt(step_val, H, W, output_dir_str):
        H, W = int(H), int(W)
        step_val = int(float(step_val))

        _output_dir, scan = _resolve_scan_context(app_state, output_dir_str)

        ckpt_list = scan["ckpt_list"]
        if not ckpt_list:
            return None, None

        steps = [s for s, _ in ckpt_list]
        step_to_path = {s: p for s, p in ckpt_list}

        closest_step = min(steps, key=lambda s: abs(s - step_val))
        path = step_to_path[closest_step]

        ckpt = _load_checkpoint_payload(path)
        scene = _scene_from_checkpoint(ckpt)

        rendered_np = _render_scene_image(scene, H, W)

        with torch.no_grad():
            gaussians = scene.get_gaussians(H, W)

        return rendered_np, {"scene_H": H, "scene_W": W, "gaussians": gaussians, "rendered_np": rendered_np}

    def inspect_pixel(x, y, state, H, W):
        H, W = int(H), int(W)
        if state is None:
            return [], None

        gaussians = state["gaussians"]
        rendered_np = state["rendered_np"]

        if gaussians is None:
            return [], rendered_np

        contributions = compute_pixel_contributions(gaussians, float(x), float(y))

        if not contributions:
            return [], rendered_np

        rows = []
        for c in contributions:
            rows.append([
                c["index"],
                c["curve_id"],
                f"{c['alpha']:.4f}",
                f"{c['color'][0]:.3f}",
                f"{c['color'][1]:.3f}",
                f"{c['color'][2]:.3f}",
                f"{c['sigma_x']:.2f}",
                f"{c['sigma_y']:.2f}",
                f"{c['rotation_deg']:.1f}",
                f"{c['mahal_sq']:.2f}",
            ])

        mask = torch.zeros(gaussians.means.shape[0], dtype=torch.bool)
        for c in contributions:
            mask[c["index"]] = True

        if mask.any():
            filtered = GaussianParams(
                means=gaussians.means[mask],
                scales=gaussians.scales[mask],
                rotations=gaussians.rotations[mask],
                colors=gaussians.colors[mask],
                opacities=gaussians.opacities[mask],
                curve_ids=gaussians.curve_ids[mask],
            )
            bg_tensor = torch.from_numpy(rendered_np).permute(2, 0, 1).float() / 255.0
            fig = render_ellipse_overlay(filtered, H, W, background=bg_tensor, max_gaussians=200)

            ax = fig.axes[0]
            ax.plot(float(x), float(y), "x", color="red", markersize=10, markeredgewidth=2)
            ax.set_title(f"Contributions at ({x:.0f}, {y:.0f})")

            overlay_np = _fig_to_numpy(fig)
        else:
            overlay_np = rendered_np

        return rows, overlay_np

    load_btn.click(
        fn=load_ckpt,
        inputs=[step_input, render_h, render_w, output_dir_state],
        outputs=[clickable_image, scene_state],
    )

    def on_image_select(state, H, W, evt: _GrSelectData):
        if evt is None or evt.index is None:
            return 0, 0, [], None
        if not isinstance(evt.index, (tuple, list)) or len(evt.index) < 2:
            return 0, 0, [], None
        x, y = evt.index[0], evt.index[1]
        rows, overlay = inspect_pixel(x, y, state, int(H), int(W))
        return x, y, rows, overlay

    clickable_image.select(
        fn=on_image_select,
        inputs=[scene_state, render_h, render_w],
        outputs=[px_x, px_y, contrib_table, overlay_image],
    )

    inspect_btn.click(
        fn=inspect_pixel,
        inputs=[px_x, px_y, scene_state, render_h, render_w],
        outputs=[contrib_table, overlay_image],
    )


def _build_pruning_tab(gr, app_state, render_h, render_w, output_dir_state):
    """Tab 4: list prune events, show before/after diff."""

    gr.Markdown("*Select a pruning event to view before/after comparison.*")

    prune_step_input = gr.Number(value=0, label="Pruning Step", precision=0)
    show_btn = gr.Button("Show Pruning Event")

    prune_image = gr.Image(label="Before / After Comparison", type="numpy")
    prune_summary = gr.Markdown()

    def show_prune_event(step_val, H, W, output_dir_str):
        H, W = int(H), int(W)
        step = int(float(step_val))

        _output_dir, scan = _resolve_scan_context(app_state, output_dir_str)

        prune_before = scan["prune_before_snaps"]
        prune_after = scan["prune_after_snaps"]

        before_by_step = {s: p for s, _n, p in prune_before}
        after_by_step = {s: p for s, _n, p in prune_after}
        prune_steps = sorted(set(before_by_step.keys()) & set(after_by_step.keys()))

        if not prune_steps:
            return None, "*No pruning events found.*"

        # Find closest prune step
        closest = min(prune_steps, key=lambda s: abs(s - step))

        pre = _load_snapshot(before_by_step[closest])
        post = _load_snapshot(after_by_step[closest])

        fig = render_prune_diff(pre, post, H, W)
        img = _fig_to_numpy(fig)

        pre_open = pre.get("n_open", 0)
        pre_closed = pre.get("n_closed", 0)
        post_open = post.get("n_open", 0)
        post_closed = post.get("n_closed", 0)

        summary = (
            f"**Step {closest} pruning event**\n\n"
            f"- Open curves: {pre_open} -> {post_open} "
            f"({'removed ' + str(pre_open - post_open) if pre_open > post_open else 'added ' + str(post_open - pre_open)})\n"
            f"- Closed curves: {pre_closed} -> {post_closed} "
            f"({'removed ' + str(pre_closed - post_closed) if pre_closed > post_closed else 'added ' + str(post_closed - pre_closed)})\n"
            f"- Total curves: {pre_open + pre_closed} -> {post_open + post_closed}\n"
            f"\nAvailable prune steps: {', '.join(str(s) for s in prune_steps)}"
        )

        return img, summary

    show_btn.click(
        fn=show_prune_event,
        inputs=[prune_step_input, render_h, render_w, output_dir_state],
        outputs=[prune_image, prune_summary],
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def launch_inspector(debug_output_dir: str | Path | None = None, **kwargs):
    """Convenience: create and launch the inspector app.

    Args:
        debug_output_dir: Path to existing debug output, or None to start fresh.
        **kwargs: Extra arguments passed to ``gr.Blocks.launch()``.
    """
    import gradio
    app = create_inspector_app(debug_output_dir)
    kwargs.setdefault("prevent_thread_lock", False)
    app.launch(theme=gradio.themes.Soft(), **kwargs)


def main():
    """CLI entry point for ``bezier-debug`` command."""
    import sys

    debug_dir = sys.argv[1] if len(sys.argv) > 1 else None
    launch_inspector(debug_dir)


if __name__ == "__main__":
    main()
