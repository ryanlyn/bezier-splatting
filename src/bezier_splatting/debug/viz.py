"""Visualization helpers for Bezier Splatting debug toolkit.

All functions return matplotlib Figure objects suitable for saving via
tracker.log_image() or displaying in the Gradio inspector.

Uses Agg backend for headless compatibility.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from ..coords import model_to_pixel
from ..model import VectorGraphicsScene
from ..rasterizer import rasterize
from ..sampling import GaussianParams


def _infer_closed_cp(state: dict[str, Tensor], fallback: int = 4) -> int:
    """Infer closed curve control-point count from serialized tensors."""
    interior = state.get("closed_interior_cp")
    if interior is not None and interior.ndim == 4:
        return int(interior.shape[2] + 2)
    return fallback


def _infer_curve_counts(state: dict[str, Tensor], n_open: int | None, n_closed: int | None) -> tuple[int, int]:
    """Infer n_open/n_closed when metadata is absent or stale."""
    open_cp = state.get("open_control_points")
    inferred_open = int(open_cp.shape[0]) if open_cp is not None else 0
    if n_open is None or (open_cp is not None and int(n_open) != inferred_open):
        n_open = inferred_open

    if "closed_shared_pts" in state:
        inferred_closed = int(state["closed_shared_pts"].shape[0])
    elif "closed_interior_cp" in state:
        inferred_closed = int(state["closed_interior_cp"].shape[0])
    else:
        inferred_closed = 0
    if n_closed is None or int(n_closed) != inferred_closed:
        n_closed = inferred_closed
    return int(n_open), int(n_closed)


def _scene_from_state(
    raw_state: dict[str, Tensor],
    n_open: int | None = None,
    n_closed: int | None = None,
    num_cp_closed_hint: int = 4,
) -> VectorGraphicsScene:
    """Reconstruct a scene from serialized state tensors."""
    state = dict(raw_state)
    n_open, n_closed = _infer_curve_counts(state, n_open, n_closed)
    num_cp_closed = _infer_closed_cp(state, fallback=num_cp_closed_hint)

    scene = VectorGraphicsScene(n_open=n_open, n_closed=n_closed, closed_cp=num_cp_closed)
    scene.load_state_dict(state, strict=False)
    return scene


def _tensor_to_numpy_image(tensor: Tensor) -> np.ndarray:
    """Convert a (3, H, W) or (H, W, 3) tensor in [0,1] to (H, W, 3) uint8 numpy."""
    t = tensor.detach().cpu().float()
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return t.clamp(0, 1).numpy()


def _fig_to_numpy(fig: Figure) -> np.ndarray:
    """Convert a matplotlib Figure to an (H, W, 3) uint8 numpy array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)
    plt.close(fig)
    return arr[:, :, :3]


def compute_error_map(rendered: Tensor, target: Tensor) -> np.ndarray:
    """Compute a colorized L1 error heatmap as (H, W, 3) uint8 numpy.

    Args:
        rendered: (3, H, W) rendered image tensor.
        target: (3, H, W) target image tensor.

    Returns:
        (H, W, 3) uint8 numpy array with hot-colormap error visualization.
    """
    import matplotlib.cm as cm

    error = (rendered.detach().cpu() - target.detach().cpu()).abs().mean(dim=0).numpy()  # (H, W)
    error_max = error.max()
    if error_max > 0:
        error_norm = error / error_max
    else:
        error_norm = error
    colored = (cm.hot(error_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def make_loss_chart(
    losses: list[float],
    psnrs: list[float],
    current_step: int,
    total_steps: int,
) -> np.ndarray:
    """Create a matplotlib loss+PSNR dual chart, returned as (H, W, 3) uint8 numpy.

    Args:
        losses: List of loss values per step.
        psnrs: List of PSNR values per step.
        current_step: Current training step (for chart title).
        total_steps: Total training steps (for chart title).

    Returns:
        (H, W, 3) uint8 numpy array of the chart.
    """
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


def _scene_from_snapshot(snapshot: dict) -> VectorGraphicsScene:
    """Reconstruct a VectorGraphicsScene from a snapshot dict (from snapshot_scene).

    The snapshot contains all parameter/buffer tensors plus n_open, n_closed metadata.
    """
    n_open = snapshot.get("n_open")
    n_closed = snapshot.get("n_closed")
    state = {key: val for key, val in snapshot.items() if isinstance(val, Tensor)}
    return _scene_from_state(state, n_open=n_open, n_closed=n_closed)


def _scene_from_checkpoint(ckpt: dict) -> VectorGraphicsScene:
    """Reconstruct a VectorGraphicsScene from a checkpoint payload dict."""
    n_open = ckpt.get("n_open")
    n_closed = ckpt.get("n_closed")
    state_dict = ckpt["state_dict"]
    num_cp_closed = int(ckpt.get("num_cp_closed", 4))
    return _scene_from_state(
        state_dict,
        n_open=n_open,
        n_closed=n_closed,
        num_cp_closed_hint=num_cp_closed,
    )


def render_gradient_heatmap(
    scene: VectorGraphicsScene,
    grad_stats: dict,
    H: int,
    W: int,
    background: Tensor | None = None,
) -> Figure:
    """Render control points as scatter, colored by gradient magnitude.

    Open curves: plot the 10 CPs as connected lines, scatter colored by |grad|.
    Closed curves: plot boundary CPs similarly.
    CPs are in [-1,1] normalized space -- scale to [0, W] and [0, H] for display.
    Uses a diverging colormap ('hot') for gradient magnitude.
    If background provided (3,H,W tensor), show it semi-transparent behind.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # flip y so (0,0) is top-left
    ax.set_aspect("equal")

    if background is not None:
        img = _tensor_to_numpy_image(background)
        ax.imshow(img, extent=[0, W, H, 0], alpha=0.4)

    # Collect all gradient magnitudes for colormap normalization
    all_mags: list[float] = []

    open_cp_grad = grad_stats.get("open_cp_grad")
    closed_cp_grad = grad_stats.get("closed_cp_grad")

    if open_cp_grad is not None:
        all_mags.extend(open_cp_grad.reshape(-1).tolist())
    if closed_cp_grad is not None:
        all_mags.extend(closed_cp_grad.reshape(-1).tolist())

    if not all_mags:
        ax.set_title("No gradients available")
        fig.tight_layout()
        return fig

    # Per-CP gradient magnitude: norm across the 2D coordinate dimension
    vmin = 0.0
    vmax = max(all_mags) if all_mags else 1.0
    if vmax < 1e-10:
        vmax = 1.0
    cmap = plt.cm.hot

    sc = None

    # Open curves
    if open_cp_grad is not None and scene.n_open > 0:
        cps = scene.open_control_points.detach().cpu()  # (N, 10, 2)
        grad_mag = open_cp_grad.cpu()  # (N, 10, 2)
        # Per-CP magnitude: norm across 2 coords
        cp_mag = torch.sqrt((grad_mag**2).sum(-1) + 1e-12)  # (N, 10)

        for i in range(scene.n_open):
            cp_px = model_to_pixel(cps[i], H, W)  # (10, 2)
            xs = cp_px[:, 0].numpy()
            ys = cp_px[:, 1].numpy()
            mags = cp_mag[i].numpy()

            ax.plot(xs, ys, "-", color="gray", linewidth=0.5, alpha=0.5)
            sc = ax.scatter(
                xs, ys, c=mags, cmap=cmap, vmin=vmin, vmax=vmax,
                s=20, edgecolors="k", linewidths=0.3, zorder=3,
            )

    # Closed curves: plot both boundaries
    if closed_cp_grad is not None and scene.n_closed > 0:
        bcp = scene.closed_boundary_cp.detach().cpu()  # (N, 2, num_cp, 2)
        grad_mag = closed_cp_grad.cpu()  # (N, 2, num_cp, 2)
        cp_mag = torch.sqrt((grad_mag**2).sum(-1) + 1e-12)  # (N, 2, num_cp)

        for i in range(scene.n_closed):
            for b in range(2):
                bcp_px = model_to_pixel(bcp[i, b], H, W)  # (num_cp, 2)
                xs = bcp_px[:, 0].numpy()
                ys = bcp_px[:, 1].numpy()
                mags = cp_mag[i, b].numpy()

                linestyle = "-" if b == 0 else "--"
                ax.plot(xs, ys, linestyle, color="gray", linewidth=0.5, alpha=0.5)
                sc = ax.scatter(
                    xs, ys, c=mags, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=20, edgecolors="k", linewidths=0.3, zorder=3,
                )

    if sc is not None:
        fig.colorbar(sc, ax=ax, label="|grad|")
    ax.set_title("Control Point Gradient Magnitudes")
    fig.tight_layout()
    return fig


def render_ellipse_overlay(
    gaussians: GaussianParams,
    H: int,
    W: int,
    background: Tensor | None = None,
    max_gaussians: int = 500,
) -> Figure:
    """Draw 3-sigma confidence ellipses for each Gaussian, color-coded by curve_id.

    Each ellipse: center at means[i], width=6*scales[i,0], height=6*scales[i,1],
    angle=degrees(rotations[i]).
    Color by curve_ids using a categorical colormap (tab20).
    Alpha=0.3 for each ellipse so overlaps are visible.
    If more than max_gaussians, subsample evenly.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")

    if background is not None:
        img = _tensor_to_numpy_image(background)
        ax.imshow(img, extent=[0, W, H, 0])

    G = gaussians.means.shape[0]
    if G == 0:
        ax.set_title("No Gaussians")
        fig.tight_layout()
        return fig

    # Subsample if needed
    if G > max_gaussians:
        indices = torch.linspace(0, G - 1, max_gaussians).long()
    else:
        indices = torch.arange(G)

    means = gaussians.means[indices].detach().cpu().numpy()
    scales = gaussians.scales[indices].detach().cpu().numpy()
    rotations = gaussians.rotations[indices].detach().cpu().float().numpy()
    opacities = torch.sigmoid(gaussians.opacities[indices]).detach().cpu().numpy()
    curve_ids = gaussians.curve_ids[indices].detach().cpu().numpy()

    unique_ids = np.unique(curve_ids)
    cmap = plt.cm.tab20
    id_to_color = {cid: cmap(i % 20) for i, cid in enumerate(unique_ids)}

    for i in range(len(indices)):
        cx, cy = means[i]
        sx, sy = scales[i]
        angle_deg = np.degrees(rotations[i])
        color = id_to_color[curve_ids[i]]
        alpha = float(opacities[i])

        ellipse = mpatches.Ellipse(
            (cx, cy),
            width=6 * sx,
            height=6 * sy,
            angle=angle_deg,
            facecolor=(*color[:3], 0.3 * alpha),
            edgecolor=(*color[:3], max(0.15, 0.7 * alpha)),
            linewidth=0.5,
        )
        ax.add_patch(ellipse)

    ax.set_title(f"Gaussian Ellipses ({len(indices)} shown, {len(unique_ids)} curves)")
    fig.tight_layout()
    return fig


def render_layer_decomposition(
    scene: VectorGraphicsScene,
    H: int,
    W: int,
    max_layers: int = 12,
) -> Figure:
    """Render each curve's contribution as a separate panel in a grid.

    For each unique curve_id (up to max_layers, prioritizing front-most):
    1. Get the full GaussianParams via scene.get_gaussians(H, W)
    2. For this curve_id, create a masked version where other curves' opacities = -100
    3. Rasterize the masked params to get this curve's solo contribution
    4. Show as a grid of images
    """
    with torch.no_grad():
        gaussians = scene.get_gaussians(H, W)

    if gaussians is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_title("No curves to decompose")
        fig.tight_layout()
        return fig

    G = gaussians.means.shape[0]
    if G == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_title("No curves to decompose")
        fig.tight_layout()
        return fig

    unique_ids = gaussians.curve_ids.unique()
    n_layers = min(len(unique_ids), max_layers)
    # Prioritize front-most: they appear first in compositing order (lowest index)
    # The Gaussians are already sorted by area (front-to-back), so take unique
    # IDs in order of first appearance.
    seen: list[int] = []
    seen_set: set[int] = set()
    ids_cpu = gaussians.curve_ids.cpu().tolist()
    for cid in ids_cpu:
        if cid not in seen_set:
            seen.append(cid)
            seen_set.add(cid)
            if len(seen) >= max_layers:
                break
    layer_ids = seen[:n_layers]

    # Grid layout
    ncols = min(4, n_layers)
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, cid in enumerate(layer_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        mask = gaussians.curve_ids == cid
        # Create masked opacities: -100 for non-matching (sigmoid(-100) ~ 0)
        masked_opacities = torch.full_like(gaussians.opacities, -100.0)
        masked_opacities[mask] = gaussians.opacities[mask]

        masked_g = GaussianParams(
            means=gaussians.means,
            scales=gaussians.scales,
            rotations=gaussians.rotations,
            colors=gaussians.colors,
            opacities=masked_opacities,
            curve_ids=gaussians.curve_ids,
        )
        with torch.no_grad():
            rendered = rasterize(masked_g, H, W)

        img = _tensor_to_numpy_image(rendered)
        ax.imshow(img, extent=[0, W, H, 0])
        ax.set_title(f"Curve {cid}", fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n_layers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.suptitle("Layer Decomposition (front-to-back)", fontsize=10)
    fig.tight_layout()
    return fig


def render_prune_diff(
    pre_snapshot: dict,
    post_snapshot: dict,
    H: int,
    W: int,
) -> Figure:
    """Side-by-side visualization of scene before/after pruning.

    Left: rendered scene before pruning.
    Right: rendered scene after pruning.
    Bottom: text summary of curves removed/added.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Reconstruct scenes from snapshots
    pre_scene = _scene_from_snapshot(pre_snapshot)
    post_scene = _scene_from_snapshot(post_snapshot)

    with torch.no_grad():
        pre_img = pre_scene.forward(H, W)
        post_img = post_scene.forward(H, W)

    axes[0].imshow(_tensor_to_numpy_image(pre_img), extent=[0, W, H, 0])
    axes[0].set_title("Before Pruning")
    axes[0].axis("off")

    axes[1].imshow(_tensor_to_numpy_image(post_img), extent=[0, W, H, 0])
    axes[1].set_title("After Pruning")
    axes[1].axis("off")

    # Summary text
    pre_open = pre_snapshot.get("n_open", 0)
    pre_closed = pre_snapshot.get("n_closed", 0)
    post_open = post_snapshot.get("n_open", 0)
    post_closed = post_snapshot.get("n_closed", 0)

    delta_open = post_open - pre_open
    delta_closed = post_closed - pre_closed
    sign = lambda x: f"+{x}" if x > 0 else str(x)

    summary = (
        f"Open: {pre_open} -> {post_open} ({sign(delta_open)})    "
        f"Closed: {pre_closed} -> {post_closed} ({sign(delta_closed)})    "
        f"Total: {pre_open + pre_closed} -> {post_open + post_closed}"
    )
    fig.text(0.5, 0.02, summary, ha="center", fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def render_training_filmstrip(
    checkpoints_dir: Path,
    H: int,
    W: int,
    max_frames: int = 8,
) -> Figure:
    """Load evenly-spaced checkpoints and render a filmstrip of training progression."""
    from .checkpoints import list_checkpoints

    ckpts = list_checkpoints(checkpoints_dir)
    if not ckpts:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, "No checkpoints found", ha="center", va="center")
        ax.axis("off")
        return fig

    # Select evenly spaced checkpoints
    n = len(ckpts)
    if n <= max_frames:
        selected = ckpts
    else:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        selected = [ckpts[i] for i in indices]

    n_frames = len(selected)
    fig, axes = plt.subplots(1, n_frames, figsize=(2.5 * n_frames, 3))
    if n_frames == 1:
        axes = [axes]

    for i, (step, path) in enumerate(selected):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        scene = _scene_from_checkpoint(ckpt)

        with torch.no_grad():
            rendered = scene.forward(H, W)

        img = _tensor_to_numpy_image(rendered)
        axes[i].imshow(img, extent=[0, W, H, 0])
        axes[i].set_title(f"Step {step}", fontsize=8)
        axes[i].axis("off")

    fig.suptitle("Training Progression", fontsize=11)
    fig.tight_layout()
    return fig
