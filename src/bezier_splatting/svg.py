"""SVG export for VectorGraphicsScene.

Control points are stored in [0, 1] normalized coordinates and
scaled to pixel space for SVG output.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .area import closed_curve_enclosed_area
from .model import VectorGraphicsScene


def _rgb_str(color: Tensor) -> str:
    """Convert (3,) float tensor in [0,1] to CSS rgb string."""
    r, g, b = (color.clamp(0, 1) * 255).int().tolist()
    return f"rgb({r},{g},{b})"


def _open_curve_to_path(
    control_points: Tensor,
    color: Tensor,
    opacity: float,
    stroke_width: float,
    H: int,
    W: int,
) -> str:
    """Convert a single open curve (10 CPs in [0,1]) to an SVG <path> element.

    3 connected cubic segments: CPs [0:4], [3:7], [6:10].
    CPs are scaled from [0,1] to pixel coordinates for SVG.
    """
    scale = torch.tensor([W, H], dtype=control_points.dtype)
    cp = control_points.detach().cpu() * scale

    # Start point
    x0, y0 = cp[0].tolist()
    d = f"M {x0:.2f},{y0:.2f}"

    # 3 cubic segments
    for seg in range(3):
        base = seg * 3
        x1, y1 = cp[base + 1].tolist()
        x2, y2 = cp[base + 2].tolist()
        x3, y3 = cp[base + 3].tolist()
        d += f" C {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}"

    color_str = _rgb_str(torch.sigmoid(color))
    sw = 0.5 + torch.sigmoid(torch.tensor(stroke_width)).item() * 4.5

    return (
        f'<path d="{d}" stroke="{color_str}" '
        f'stroke-width="{sw:.2f}" fill="none" '
        f'opacity="{opacity:.3f}" stroke-linecap="round"/>'
    )


def _closed_curve_to_path(
    boundary_cp: Tensor,
    color: Tensor,
    opacity: float,
    H: int,
    W: int,
) -> str:
    """Convert a closed curve (paired boundaries in [0,1]) to SVG filled <path>.

    Draws the top boundary forward, then the bottom boundary backward, and closes.
    CPs are scaled from [0,1] to pixel coordinates.
    """
    scale = torch.tensor([W, H], dtype=boundary_cp.dtype)
    top_cp = boundary_cp[0].detach().cpu() * scale  # (num_cp, 2)
    bot_cp = boundary_cp[1].detach().cpu() * scale

    num_cp = top_cp.shape[0]

    # Top boundary: forward
    x0, y0 = top_cp[0].tolist()
    d = f"M {x0:.2f},{y0:.2f}"

    if num_cp == 4:
        # Single cubic
        x1, y1 = top_cp[1].tolist()
        x2, y2 = top_cp[2].tolist()
        x3, y3 = top_cp[3].tolist()
        d += f" C {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}"
    else:
        for i in range(1, num_cp):
            x, y = top_cp[i].tolist()
            d += f" L {x:.2f},{y:.2f}"

    # Line to bottom boundary end
    xb, yb = bot_cp[-1].tolist()
    d += f" L {xb:.2f},{yb:.2f}"

    # Bottom boundary: backward
    if num_cp == 4:
        x1, y1 = bot_cp[2].tolist()
        x2, y2 = bot_cp[1].tolist()
        x3, y3 = bot_cp[0].tolist()
        d += f" C {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}"
    else:
        for i in range(num_cp - 2, -1, -1):
            x, y = bot_cp[i].tolist()
            d += f" L {x:.2f},{y:.2f}"

    d += " Z"

    color_str = _rgb_str(torch.sigmoid(color))

    return f'<path d="{d}" fill="{color_str}" opacity="{opacity:.3f}"/>'


def scene_to_svg(scene: VectorGraphicsScene, H: int | None = None, W: int | None = None) -> str:
    """Export a VectorGraphicsScene as an SVG string.

    Args:
        scene: Trained scene model.
        H, W: SVG dimensions. Defaults to scene's H, W.

    Returns:
        SVG string.
    """
    H = H or scene.H
    W = W or scene.W

    elements: list[tuple[float, str]] = []  # (area, svg_element)

    # Closed curves (usually larger → background)
    if scene.n_closed > 0:
        closed_opacities = torch.sigmoid(scene.closed_opacities).detach().cpu()
        scale = torch.tensor([W, H], dtype=torch.float32)
        for i in range(scene.n_closed):
            opacity = closed_opacities[i].item()
            if opacity < 0.01:
                continue
            bcp = scene.closed_boundary_cp[i]
            bcp_px = bcp.detach().cpu() * scale
            # True enclosed area (not bounding box)
            area = closed_curve_enclosed_area(bcp_px.unsqueeze(0))[0].item()
            svg_elem = _closed_curve_to_path(bcp, scene.closed_colors[i], opacity, H, W)
            elements.append((area, svg_elem))

    # Open curves
    if scene.n_open > 0:
        # Per-segment opacity → mean for SVG (SVG has no per-segment opacity)
        open_opacities = torch.sigmoid(scene.open_opacities).detach().cpu()  # (N, 3)
        mean_opacity = open_opacities.mean(dim=-1)  # (N,)
        scale = torch.tensor([W, H], dtype=torch.float32)
        for i in range(scene.n_open):
            opacity = mean_opacity[i].item()
            if opacity < 0.01:
                continue
            cp = scene.open_control_points[i]
            cp_px = cp.detach().cpu() * scale
            edge_len = torch.norm(cp_px[1:] - cp_px[:-1], dim=-1).sum().item()
            sw = 0.5 + torch.sigmoid(scene.open_stroke_widths[i]).item() * 4.5
            area = edge_len * sw
            svg_elem = _open_curve_to_path(
                cp, scene.open_colors[i], opacity, scene.open_stroke_widths[i].item(),
                H, W,
            )
            elements.append((area, svg_elem))

    # Sort: larger area first (background), smaller on top (foreground)
    elements.sort(key=lambda x: x[0], reverse=True)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
    ]
    for _, elem in elements:
        svg_parts.append(f"  {elem}")
    svg_parts.append("</svg>")

    return "\n".join(svg_parts)
