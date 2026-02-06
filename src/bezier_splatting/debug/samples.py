"""Sample target images for the Bezier Splatting debug toolkit.

Provides programmatic targets (circle, overlap, strokes, gradient, composition),
external image loading, and Kodak sample image discovery.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Programmatic targets
# ---------------------------------------------------------------------------


def generate_circle(H: int = 256, W: int = 256) -> Tensor:
    """Red circle on white background. Returns (3, H, W) tensor."""
    img = torch.ones(3, H, W)
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy, r = W / 2, H / 2, min(H, W) / 4
    mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
    img[0, mask] = 1.0
    img[1, mask] = 0.0
    img[2, mask] = 0.0
    return img


def generate_overlap(H: int = 256, W: int = 256) -> Tensor:
    """3 semi-transparent colored rectangles. Returns (3, H, W) tensor."""
    img = torch.ones(3, H, W)

    s = H / 256
    rects = [
        (int(40 * s), int(40 * s), int(160 * s), int(160 * s), torch.tensor([1.0, 0.0, 0.0]), 0.5),
        (int(80 * s), int(80 * s), int(200 * s), int(200 * s), torch.tensor([0.0, 1.0, 0.0]), 0.5),
        (int(60 * s), int(100 * s), int(180 * s), int(220 * s), torch.tensor([0.0, 0.0, 1.0]), 0.5),
    ]

    for y0, x0, y1, x1, color, alpha in rects:
        for c in range(3):
            img[c, y0:y1, x0:x1] = img[c, y0:y1, x0:x1] * (1 - alpha) + color[c] * alpha

    return img


def generate_strokes(H: int = 256, W: int = 256) -> Tensor:
    """Star pattern with 8 radiating lines. Returns (3, H, W) tensor."""
    img = torch.ones(3, H, W)
    cx, cy = W / 2, H / 2
    length = min(H, W) * 0.4

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    for i in range(8):
        angle = i * math.pi / 4
        dx = math.cos(angle)
        dy = math.sin(angle)
        thickness = 1.0 + (i % 4)

        dist_to_line = abs((x_coords - cx) * dy - (y_coords - cy) * dx)
        proj = (x_coords - cx) * dx + (y_coords - cy) * dy
        on_segment = (proj >= 0) & (proj <= length)

        mask = (dist_to_line < thickness) & on_segment
        img[0, mask] = 0.2
        img[1, mask] = 0.2
        img[2, mask] = 0.2

    return img


def generate_gradient(H: int = 256, W: int = 256) -> Tensor:
    """Linear color gradient (blue to orange). Returns (3, H, W) tensor."""
    t = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    img = torch.zeros(3, H, W)

    blue = torch.tensor([0.1, 0.2, 0.8])
    orange = torch.tensor([1.0, 0.6, 0.1])

    for c in range(3):
        img[c] = blue[c] * (1 - t) + orange[c] * t

    return img


def generate_composition(H: int = 256, W: int = 256) -> Tensor:
    """Multi-element scene with shapes, strokes, and gradient background. Returns (3, H, W) tensor."""
    img = generate_gradient(H, W)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    s = H / 256

    shapes = [
        ((64 * s, 64 * s), 30 * s, torch.tensor([0.9, 0.1, 0.1])),
        ((192 * s, 64 * s), 25 * s, torch.tensor([0.1, 0.9, 0.1])),
        ((128 * s, 192 * s), 35 * s, torch.tensor([0.9, 0.9, 0.1])),
    ]
    for (cy, cx), r, color in shapes:
        mask = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) < r ** 2
        alpha = 0.8
        for c in range(3):
            img[c, mask] = img[c, mask] * (1 - alpha) + color[c] * alpha

    for i in range(5):
        y0 = (30 + i * 40) * s
        x0 = 20 * s
        x1 = W - 20 * s
        thickness = max(1.0, 2.0 * s)
        mask = (
            (y_coords >= y0 - thickness)
            & (y_coords <= y0 + thickness)
            & (x_coords >= x0)
            & (x_coords <= x1)
        )
        gray = 0.1 + i * 0.15
        for c in range(3):
            img[c, mask] = gray

    return img.clamp(0, 1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_sample_targets() -> dict[str, Callable[..., Tensor]]:
    """Return dict mapping display name -> generator function."""
    return {
        "Circle (closed curves)": generate_circle,
        "Overlapping Rectangles (depth + alpha)": generate_overlap,
        "Star Strokes (open curves)": generate_strokes,
        "Color Gradient (smooth fills)": generate_gradient,
        "Composition (mixed curves)": generate_composition,
    }


SUGGESTED_PARAMS: dict[str, dict] = {
    "Circle (closed curves)": {"n_open": 0, "n_closed": 8, "steps": 1000},
    "Overlapping Rectangles (depth + alpha)": {"n_open": 0, "n_closed": 16, "steps": 1500},
    "Star Strokes (open curves)": {"n_open": 32, "n_closed": 0, "steps": 2000},
    "Color Gradient (smooth fills)": {"n_open": 16, "n_closed": 16, "steps": 1500},
    "Composition (mixed curves)": {"n_open": 32, "n_closed": 16, "steps": 2000},
}


# ---------------------------------------------------------------------------
# External image loading
# ---------------------------------------------------------------------------


def load_image(path: str | Path, H: int = 256, W: int = 256) -> Tensor:
    """Load an image file, resize to (H, W), return (3, H, W) float tensor in [0, 1].

    Handles RGBA by compositing onto white. Handles grayscale by expanding to RGB.
    """
    from PIL import Image

    img = Image.open(path)

    # Convert RGBA to RGB (composite onto white)
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[[-1]])
        img = background
    elif img.mode == "LA" or img.mode == "P":
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize with bilinear interpolation
    img = img.resize((W, H), Image.BILINEAR)

    # Convert to tensor (3, H, W) in [0, 1]
    import numpy as np

    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


# ---------------------------------------------------------------------------
# Kodak sample images
# ---------------------------------------------------------------------------

_KODAK_SAMPLES = {
    "Lighthouse (kodim04)": "kodim04.png",
    "House (kodim07)": "kodim07.png",
    "Parrot (kodim23)": "kodim23.png",
    "Rocks (kodim08)": "kodim08.png",
}


def _default_samples_dir() -> Path:
    """Return the default samples/ directory at the project root."""
    # Walk up from this file to find the project root (contains pyproject.toml)
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current / "samples"
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent.parent / "samples"


def get_kodak_samples(samples_dir: Path | None = None) -> dict[str, Path]:
    """Return dict of available Kodak sample image paths.

    Looks in samples_dir (default: project_root/samples/).
    Returns only images that actually exist on disk.
    """
    if samples_dir is None:
        samples_dir = _default_samples_dir()
    samples_dir = Path(samples_dir)

    available: dict[str, Path] = {}
    for display_name, filename in _KODAK_SAMPLES.items():
        path = samples_dir / filename
        if path.exists():
            available[display_name] = path

    return available
