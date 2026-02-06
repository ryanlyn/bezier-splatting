"""Reconstruction evaluation suite with programmatic test targets.

Each target tests a specific aspect of the pipeline. Targets are generated
programmatically (no external images needed).

Thresholds are tiered:
    - Tier 1 (must pass): PSNR > 20, SSIM > 0.7 — if these fail, something fundamental is broken
    - Tier 2 (quality gate): PSNR > 28, SSIM > 0.85 — "good enough" reconstruction
"""

import json
from pathlib import Path

import torch
import pytest

from bezier_splatting.debug.samples import (
    generate_circle,
    generate_composition,
    generate_gradient,
    generate_overlap,
    generate_strokes,
)
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

# Subset for --fast mode: one closed-only, one open-only (covers both samplers)
FAST_TARGETS = ["circle", "strokes"]


# ── Programmatic target generation (imported from debug.samples) ──

GENERATORS = {
    "circle": generate_circle,
    "overlap": generate_overlap,
    "strokes": generate_strokes,
    "gradient": generate_gradient,
    "composition": generate_composition,
}


# ── Optimization cache ──
# Cache fit_image() results so tier1 and tier2 share the same run per target.
_optimization_cache: dict[str, tuple] = {}


# ── Test fixtures ──

@pytest.fixture
def save_outputs(request):
    return request.config.getoption("--save-outputs", default=False)


@pytest.fixture(params=list(TARGET_CONFIGS.keys()))
def target_config(request):
    """Yield (target_name, target_image, config) for each test target."""
    name = request.param
    fast = request.config.getoption("--fast", default=False)

    if fast and name not in FAST_TARGETS:
        pytest.skip("skipped in --fast mode")

    config = TARGET_CONFIGS[name].copy()
    if fast:
        config["steps"] = config["steps"] // 2

    # Fixed seed per target for reproducible results across runs
    torch.manual_seed(hash(name) % 2**32)

    target = GENERATORS[name]()
    return name, target, config


# ── Tests ──

class TestReconstruction:
    def _get_optimized(self, name, target, config):
        """Run or retrieve cached optimization for a target."""
        if name not in _optimization_cache:
            scene = fit_image(
                target,
                n_open=config["n_open"],
                n_closed=config["n_closed"],
                steps=config["steps"],
                log_every=config["steps"] // 10,
            )
            rendered = scene(target.shape[1], target.shape[2]).detach()
            metrics = compute_metrics(rendered, target)
            _optimization_cache[name] = (scene, rendered, metrics)
        return _optimization_cache[name]

    @pytest.mark.slow
    def test_reconstruction_tier1(self, target_config, save_outputs):
        """Tier 1: Sanity check. If this fails, something fundamental is broken."""
        name, target, config = target_config
        scene, rendered, metrics = self._get_optimized(name, target, config)

        if save_outputs:
            _save_diagnostics(name, target, rendered, scene, metrics)

        assert metrics["psnr"] > config["tier1_psnr"], (
            f"[{name}] PSNR {metrics['psnr']:.1f} below tier-1 minimum {config['tier1_psnr']}"
        )
        assert metrics["ssim"] > config["tier1_ssim"], (
            f"[{name}] SSIM {metrics['ssim']:.3f} below tier-1 minimum {config['tier1_ssim']}"
        )

    @pytest.mark.slow
    def test_reconstruction_tier2(self, target_config, save_outputs, request):
        """Tier 2: Quality gate. Good enough reconstruction."""
        if request.config.getoption("--fast", default=False):
            pytest.skip("tier-2 skipped in --fast mode")

        name, target, config = target_config
        scene, rendered, metrics = self._get_optimized(name, target, config)

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
