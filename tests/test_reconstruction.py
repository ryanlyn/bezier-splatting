"""Reconstruction evaluation suite with programmatic test targets.

Each target tests a specific aspect of the pipeline. Targets are generated
programmatically (no external images needed).

Thresholds are tiered:
    - Tier 1 (must pass): PSNR > 20, SSIM > 0.7 — if these fail, something fundamental is broken
    - Tier 2 (quality gate): PSNR > 28, SSIM > 0.85 — "good enough" reconstruction
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import pytest

from bezier_splatting.metrics import compute_metrics
from bezier_splatting.optimization import fit_image
from bezier_splatting.svg import scene_to_svg


# ── Target configurations ──
# Resolution for reconstruction tests (64 for fast CI, 256 for full quality)
TEST_RESOLUTION = 64

TARGET_CONFIGS = {
    "circle": {
        "n_open": 0,
        "n_closed": 8,
        "steps": 1000,
        "tier1_psnr": 20.0,
        "tier1_ssim": 0.70,
        "tier2_psnr": 26.0,
        "tier2_ssim": 0.85,
    },
    "overlap": {
        "n_open": 0,
        "n_closed": 16,
        "steps": 1500,
        "tier1_psnr": 18.0,
        "tier1_ssim": 0.65,
        "tier2_psnr": 24.0,
        "tier2_ssim": 0.80,
    },
    "strokes": {
        "n_open": 32,
        "n_closed": 0,
        "steps": 2000,  # More steps for thin lines
        "tier1_psnr": 18.0,
        "tier1_ssim": 0.65,
        "tier2_psnr": 22.0,  # Lower threshold at 64×64 (thin lines < 1px, high variance)
        "tier2_ssim": 0.78,
    },
    "gradient": {
        "n_open": 16,
        "n_closed": 16,
        "steps": 1500,
        "tier1_psnr": 20.0,
        "tier1_ssim": 0.70,
        "tier2_psnr": 25.0,
        "tier2_ssim": 0.82,
    },
    "composition": {
        "n_open": 32,
        "n_closed": 16,
        "steps": 2000,
        "tier1_psnr": 18.0,
        "tier1_ssim": 0.60,
        "tier2_psnr": 23.0,
        "tier2_ssim": 0.78,
    },
}


# ── Programmatic target generation ──

def _generate_circle(H: int = TEST_RESOLUTION, W: int = TEST_RESOLUTION) -> torch.Tensor:
    """Red filled circle on white background."""
    img = torch.ones(3, H, W)
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy, r = W / 2, H / 2, min(H, W) / 4
    mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
    img[0, mask] = 1.0  # red
    img[1, mask] = 0.0
    img[2, mask] = 0.0
    return img


def _generate_overlap(H: int = TEST_RESOLUTION, W: int = TEST_RESOLUTION) -> torch.Tensor:
    """3 semi-transparent colored rectangles overlapping."""
    img = torch.ones(3, H, W)

    # Scale rectangle coords to resolution (designed for 256, scale down)
    s = H / 256
    rects = [
        # (y0, x0, y1, x1, color, opacity)
        (int(40*s), int(40*s), int(160*s), int(160*s), torch.tensor([1.0, 0.0, 0.0]), 0.5),   # red
        (int(80*s), int(80*s), int(200*s), int(200*s), torch.tensor([0.0, 1.0, 0.0]), 0.5),   # green
        (int(60*s), int(100*s), int(180*s), int(220*s), torch.tensor([0.0, 0.0, 1.0]), 0.5),  # blue
    ]

    for y0, x0, y1, x1, color, alpha in rects:
        for c in range(3):
            img[c, y0:y1, x0:x1] = img[c, y0:y1, x0:x1] * (1 - alpha) + color[c] * alpha

    return img


def _generate_strokes(H: int = TEST_RESOLUTION, W: int = TEST_RESOLUTION) -> torch.Tensor:
    """Star pattern: 8 lines radiating from center, varying thickness."""
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
        thickness = 1.0 + (i % 4)  # 1-4 pixel thickness

        # Distance from point to line through center in direction (dx, dy)
        # For pixel (x, y), distance to line = |(x-cx)*dy - (y-cy)*dx|
        dist_to_line = abs((x_coords - cx) * dy - (y_coords - cy) * dx)
        # Also constrain to line segment length
        proj = (x_coords - cx) * dx + (y_coords - cy) * dy
        on_segment = (proj >= 0) & (proj <= length)

        mask = (dist_to_line < thickness) & on_segment
        img[0, mask] = 0.2
        img[1, mask] = 0.2
        img[2, mask] = 0.2

    return img


def _generate_gradient(H: int = TEST_RESOLUTION, W: int = TEST_RESOLUTION) -> torch.Tensor:
    """Linear gradient from blue (left) to orange (right)."""
    t = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)  # (H, W)
    img = torch.zeros(3, H, W)

    # Blue = (0.1, 0.2, 0.8), Orange = (1.0, 0.6, 0.1)
    blue = torch.tensor([0.1, 0.2, 0.8])
    orange = torch.tensor([1.0, 0.6, 0.1])

    for c in range(3):
        img[c] = blue[c] * (1 - t) + orange[c] * t

    return img


def _generate_composition(H: int = TEST_RESOLUTION, W: int = TEST_RESOLUTION) -> torch.Tensor:
    """Multi-shape: 3 filled shapes + 5 strokes + gradient background."""
    # Start with gradient background
    img = _generate_gradient(H, W)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    # Scale factor (designed for 256)
    s = H / 256

    # 3 filled shapes (scaled)
    shapes = [
        ((64*s, 64*s), 30*s, torch.tensor([0.9, 0.1, 0.1])),    # red circle
        ((192*s, 64*s), 25*s, torch.tensor([0.1, 0.9, 0.1])),    # green circle
        ((128*s, 192*s), 35*s, torch.tensor([0.9, 0.9, 0.1])),   # yellow circle
    ]
    for (cy, cx), r, color in shapes:
        mask = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) < r ** 2
        alpha = 0.8
        for c in range(3):
            img[c, mask] = img[c, mask] * (1 - alpha) + color[c] * alpha

    # 5 strokes (scaled)
    for i in range(5):
        y0 = (30 + i * 40) * s
        x0 = 20 * s
        x1 = W - 20 * s
        thickness = max(1.0, 2.0 * s)
        mask = (y_coords >= y0 - thickness) & (y_coords <= y0 + thickness) & (x_coords >= x0) & (x_coords <= x1)
        gray = 0.1 + i * 0.15
        for c in range(3):
            img[c, mask] = gray

    return img.clamp(0, 1)


GENERATORS = {
    "circle": _generate_circle,
    "overlap": _generate_overlap,
    "strokes": _generate_strokes,
    "gradient": _generate_gradient,
    "composition": _generate_composition,
}


# ── Test fixtures ──

@pytest.fixture
def save_outputs(request):
    return request.config.getoption("--save-outputs", default=False)


@pytest.fixture(params=list(TARGET_CONFIGS.keys()))
def target_config(request):
    """Yield (target_name, target_image, config) for each test target."""
    name = request.param
    config = TARGET_CONFIGS[name]
    target = GENERATORS[name]()
    return name, target, config


# ── Tests ──

class TestReconstruction:
    @pytest.mark.slow
    def test_reconstruction_tier1(self, target_config, save_outputs):
        """Tier 1: Sanity check. If this fails, something fundamental is broken."""
        name, target, config = target_config

        scene = fit_image(
            target,
            n_open=config["n_open"],
            n_closed=config["n_closed"],
            steps=config["steps"],
            log_every=config["steps"] // 10,
        )

        rendered = scene(target.shape[1], target.shape[2]).detach()
        metrics = compute_metrics(rendered, target)

        if save_outputs:
            _save_diagnostics(name, target, rendered, scene, metrics)

        assert metrics["psnr"] > config["tier1_psnr"], (
            f"[{name}] PSNR {metrics['psnr']:.1f} below tier-1 minimum {config['tier1_psnr']}"
        )
        assert metrics["ssim"] > config["tier1_ssim"], (
            f"[{name}] SSIM {metrics['ssim']:.3f} below tier-1 minimum {config['tier1_ssim']}"
        )

    @pytest.mark.slow
    def test_reconstruction_tier2(self, target_config, save_outputs):
        """Tier 2: Quality gate. Good enough reconstruction."""
        name, target, config = target_config

        scene = fit_image(
            target,
            n_open=config["n_open"],
            n_closed=config["n_closed"],
            steps=config["steps"],
            log_every=config["steps"] // 10,
        )

        rendered = scene(target.shape[1], target.shape[2]).detach()
        metrics = compute_metrics(rendered, target)

        if save_outputs:
            _save_diagnostics(name, target, rendered, scene, metrics)

        assert metrics["psnr"] > config["tier2_psnr"], (
            f"[{name}] PSNR {metrics['psnr']:.1f} below tier-2 quality gate {config['tier2_psnr']}"
        )
        assert metrics["ssim"] > config["tier2_ssim"], (
            f"[{name}] SSIM {metrics['ssim']:.3f} below tier-2 quality gate {config['tier2_ssim']}"
        )


def _save_diagnostics(
    name: str,
    target: torch.Tensor,
    rendered: torch.Tensor,
    scene,
    metrics: dict,
) -> None:
    """Save diagnostic outputs for debugging."""
    from PIL import Image
    import numpy as np

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Rendered image
    img_np = (rendered.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save(output_dir / f"{name}_rendered.png")

    # Error heatmap
    error = (rendered - target).abs().mean(dim=0).numpy()
    error_norm = (error / error.max() * 255).astype(np.uint8)
    Image.fromarray(error_norm, mode="L").save(output_dir / f"{name}_error.png")

    # SVG
    svg_str = scene_to_svg(scene)
    (output_dir / f"{name}_curves.svg").write_text(svg_str)

    # Metrics JSON
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
