"""Tests for topology.py — pruning/densification heuristics."""

import torch
import pytest

from bezier_splatting.topology import (
    PruneConfig,
    compute_aabb,
    compute_outside_ratio,
    compute_pairwise_iou,
    compute_color_distance,
    compute_tiny_curve_mask,
    compute_overlap_suppression_mask,
    compute_prune_mask_open,
    compute_prune_mask_closed,
    compute_densify_centers,
)
from bezier_splatting.model import VectorGraphicsScene


# ── compute_aabb ─────────────────────────────────────────────────────────


class TestComputeAABB:
    """Test axis-aligned bounding box computation."""

    def test_known_cps(self):
        """Control points at known model-space positions produce expected AABB."""
        H, W = 64, 64
        # Single curve with 4 CPs spanning [-1, 1] in x, [0, 0] in y
        # model_to_pixel: (cp + 1) / 2 * [W, H]
        # x=-1 -> px 0, x=1 -> px 64, y=0 -> px 32
        cp = torch.tensor([
            [[-1.0, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        ])  # (1, 4, 2)
        aabb = compute_aabb(cp, H, W)
        assert aabb.shape == (1, 4)
        # x_min=0, y_min=32, x_max=64, y_max=32
        torch.testing.assert_close(aabb[0, 0], torch.tensor(0.0), atol=1e-5, rtol=0)
        torch.testing.assert_close(aabb[0, 2], torch.tensor(64.0), atol=1e-5, rtol=0)
        assert aabb[0, 1] == aabb[0, 3]  # all y=0 in model -> same pixel y

    def test_multiple_curves(self):
        """Multiple curves each get their own AABB."""
        H, W = 100, 100
        cp = torch.tensor([
            [[-1.0, -1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.5, 0.5]],
        ])  # (2, 2, 2)
        aabb = compute_aabb(cp, H, W)
        assert aabb.shape == (2, 4)
        # Curve 0: spans full image
        assert aabb[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert aabb[0, 1].item() == pytest.approx(0.0, abs=1e-5)
        assert aabb[0, 2].item() == pytest.approx(100.0, abs=1e-5)
        assert aabb[0, 3].item() == pytest.approx(100.0, abs=1e-5)

    def test_empty(self):
        """Empty input returns empty output."""
        aabb = compute_aabb(torch.empty(0, 4, 2), 64, 64)
        assert aabb.shape == (0, 4)


# ── compute_outside_ratio ────────────────────────────────────────────────


class TestComputeOutsideRatio:
    """Test outside-image fraction computation."""

    def test_fully_inside(self):
        """A box fully inside the image has ratio 0."""
        aabb = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        ratio = compute_outside_ratio(aabb, 64, 64)
        assert ratio.shape == (1,)
        assert ratio[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_fully_outside(self):
        """A box fully outside the image has ratio 1."""
        aabb = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        ratio = compute_outside_ratio(aabb, 64, 64)
        assert ratio[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_half_outside(self):
        """A box half-outside has ratio ~0.5."""
        # Box from x=-32..32, y=0..64 in a 64x64 image
        # Clipped area = 32*64 = 2048, total area = 64*64 = 4096
        aabb = torch.tensor([[-32.0, 0.0, 32.0, 64.0]])
        ratio = compute_outside_ratio(aabb, 64, 64)
        assert ratio[0].item() == pytest.approx(0.5, abs=1e-5)

    def test_empty(self):
        """Empty input returns empty output."""
        ratio = compute_outside_ratio(torch.empty(0, 4), 64, 64)
        assert ratio.shape == (0,)


# ── compute_pairwise_iou ────────────────────────────────────────────────


class TestComputePairwiseIoU:
    """Test pairwise IoU between bounding boxes."""

    def test_identical_boxes(self):
        """Identical boxes have IoU = 1 (off-diagonal) and 0 on diagonal."""
        aabb = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ])
        iou = compute_pairwise_iou(aabb)
        assert iou.shape == (2, 2)
        assert iou[0, 0].item() == 0.0  # diagonal
        assert iou[0, 1].item() == pytest.approx(1.0, abs=1e-5)
        assert iou[1, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_non_overlapping(self):
        """Non-overlapping boxes have IoU = 0."""
        aabb = torch.tensor([
            [0.0, 0.0, 5.0, 5.0],
            [10.0, 10.0, 20.0, 20.0],
        ])
        iou = compute_pairwise_iou(aabb)
        assert iou[0, 1].item() == pytest.approx(0.0, abs=1e-5)

    def test_known_overlap(self):
        """Boxes with known overlap produce expected IoU."""
        # Box A: [0,0,10,10] area=100
        # Box B: [5,0,15,10] area=100
        # Intersection: [5,0,10,10] area=50
        # Union: 100+100-50=150
        # IoU = 50/150 = 1/3
        aabb = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 0.0, 15.0, 10.0],
        ])
        iou = compute_pairwise_iou(aabb)
        assert iou[0, 1].item() == pytest.approx(1.0 / 3.0, abs=1e-5)

    def test_empty(self):
        """Empty input returns empty matrix."""
        iou = compute_pairwise_iou(torch.empty(0, 4))
        assert iou.shape == (0, 0)

    def test_symmetry(self):
        """IoU matrix is symmetric."""
        aabb = torch.rand(5, 4)
        aabb[:, 2] = aabb[:, 0] + torch.rand(5) * 10 + 1
        aabb[:, 3] = aabb[:, 1] + torch.rand(5) * 10 + 1
        iou = compute_pairwise_iou(aabb)
        torch.testing.assert_close(iou, iou.T)


# ── compute_tiny_curve_mask ──────────────────────────────────────────────


class TestComputeTinyCurveMask:
    """Test tiny curve detection."""

    def test_large_curves_kept(self):
        """Curves exceeding all thresholds are kept."""
        aabb = torch.tensor([
            [0.0, 0.0, 20.0, 20.0],  # w=20, h=20, area=400
        ])
        mask = compute_tiny_curve_mask(aabb, 5.0, 5.0, 16.0)
        assert mask[0].item() is True

    def test_tiny_curves_pruned(self):
        """Curves below all three thresholds are pruned."""
        aabb = torch.tensor([
            [0.0, 0.0, 2.0, 2.0],  # w=2, h=2, area=4
        ])
        mask = compute_tiny_curve_mask(aabb, 5.0, 5.0, 16.0)
        assert mask[0].item() is False

    def test_only_width_small(self):
        """Curve small in width only is kept (all three must be below)."""
        aabb = torch.tensor([
            [0.0, 0.0, 3.0, 20.0],  # w=3, h=20, area=60
        ])
        mask = compute_tiny_curve_mask(aabb, 5.0, 5.0, 16.0)
        assert mask[0].item() is True  # h and area are above thresholds

    def test_empty(self):
        """Empty input returns empty mask."""
        mask = compute_tiny_curve_mask(torch.empty(0, 4), 5.0, 5.0, 16.0)
        assert mask.shape == (0,)


# ── compute_overlap_suppression_mask ─────────────────────────────────────


class TestComputeOverlapSuppressionMask:
    """Test overlap + color similarity suppression."""

    def test_overlapping_same_color_smaller_pruned(self):
        """Two overlapping same-color curves: the smaller one gets suppressed."""
        # Large box and small box that overlap significantly
        aabb = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],   # area=10000
            [10.0, 10.0, 30.0, 30.0],    # area=400, inside the large one
        ])
        # Same pre-sigmoid colors (both will sigmoid to ~0.5)
        colors = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        areas = torch.tensor([10000.0, 400.0])
        mask = compute_overlap_suppression_mask(
            aabb, colors, areas,
            iou_threshold=0.01,
            color_threshold=0.1,
            weighted_iou_threshold=0.01,
            area_threshold=50000.0,
        )
        assert mask[0].item() is True   # large one kept
        assert mask[1].item() is False  # small one suppressed

    def test_overlapping_different_color_kept(self):
        """Overlapping curves with very different colors are both kept."""
        aabb = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
            [10.0, 10.0, 30.0, 30.0],
        ])
        # Very different colors
        colors = torch.tensor([
            [-5.0, -5.0, -5.0],  # sigmoid -> ~0.007
            [5.0, 5.0, 5.0],     # sigmoid -> ~0.993
        ])
        areas = torch.tensor([10000.0, 400.0])
        mask = compute_overlap_suppression_mask(
            aabb, colors, areas,
            iou_threshold=0.01,
            color_threshold=0.05,
            weighted_iou_threshold=0.01,
            area_threshold=50000.0,
        )
        assert mask.all()  # both kept

    def test_non_overlapping_kept(self):
        """Non-overlapping same-color curves are both kept."""
        aabb = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [50.0, 50.0, 60.0, 60.0],
        ])
        colors = torch.zeros(2, 3)
        areas = torch.tensor([100.0, 100.0])
        mask = compute_overlap_suppression_mask(
            aabb, colors, areas,
            iou_threshold=0.01,
            color_threshold=0.1,
            weighted_iou_threshold=0.01,
            area_threshold=50000.0,
        )
        assert mask.all()


# ── compute_prune_mask_open ──────────────────────────────────────────────


class TestComputePruneMaskOpen:
    """Test open curve pruning masks."""

    def test_returns_correct_shape(self):
        """Mask shape matches number of open curves."""
        scene = VectorGraphicsScene(n_open=5, n_closed=0, H=64, W=64)
        config = PruneConfig()
        mask, metrics = compute_prune_mask_open(scene, 0.5, config, 64, 64)
        assert mask.shape == (5,)
        assert len(metrics) == 5

    def test_empty_scene(self):
        """Empty scene returns empty mask."""
        scene = VectorGraphicsScene(n_open=0, n_closed=0, H=64, W=64)
        config = PruneConfig()
        mask, metrics = compute_prune_mask_open(scene, 0.5, config, 64, 64)
        assert mask.shape == (0,)
        assert len(metrics) == 0

    def test_metrics_have_expected_keys(self):
        """Per-curve metrics contain the expected diagnostic keys."""
        scene = VectorGraphicsScene(n_open=3, n_closed=0, H=64, W=64)
        config = PruneConfig()
        _, metrics = compute_prune_mask_open(scene, 0.5, config, 64, 64)
        expected_keys = {"outside_ratio", "max_iou", "opacity", "area", "pruned", "reason"}
        for m in metrics:
            assert set(m.keys()) == expected_keys


# ── compute_prune_mask_closed ────────────────────────────────────────────


class TestComputePruneMaskClosed:
    """Test closed curve pruning masks."""

    def test_returns_correct_shape(self):
        """Mask shape matches number of closed curves."""
        scene = VectorGraphicsScene(n_open=0, n_closed=4, H=64, W=64)
        config = PruneConfig()
        mask, metrics = compute_prune_mask_closed(scene, 0.5, config, 64, 64)
        assert mask.shape == (4,)
        assert len(metrics) == 4

    def test_empty_scene(self):
        """Empty scene returns empty mask."""
        scene = VectorGraphicsScene(n_open=0, n_closed=0, H=64, W=64)
        config = PruneConfig()
        mask, metrics = compute_prune_mask_closed(scene, 0.5, config, 64, 64)
        assert mask.shape == (0,)
        assert len(metrics) == 0

    def test_staged_opacity_threshold(self):
        """Closed curve pruning uses staged opacity thresholds."""
        scene = VectorGraphicsScene(n_open=0, n_closed=3, H=64, W=64)
        config = PruneConfig(
            opacity_threshold_closed_early=0.01,
            opacity_threshold_closed_late=0.99,
        )
        # At early progress (< 0.7), the threshold is very low -> keep more
        mask_early, _ = compute_prune_mask_closed(scene, 0.3, config, 64, 64)
        # At late progress (>= 0.7), the threshold is very high -> prune more
        mask_late, _ = compute_prune_mask_closed(scene, 0.8, config, 64, 64)
        # Late should prune at least as many as early (usually more)
        assert mask_late.sum() <= mask_early.sum()


# ── compute_densify_centers ──────────────────────────────────────────────


class TestComputeDensifyCenters:
    """Test error-hotspot center computation."""

    def test_returns_pixel_centers(self):
        """Centers are in pixel space and have correct shape."""
        rendered = torch.zeros(3, 64, 64)
        target = torch.ones(3, 64, 64)
        centers = compute_densify_centers(rendered, target, 3, 64, 64)
        assert centers.shape[0] <= 3
        assert centers.shape[1] == 2
        # Centers should be within image bounds
        assert (centers[:, 0] >= 0).all()
        assert (centers[:, 0] <= 64).all()
        assert (centers[:, 1] >= 0).all()
        assert (centers[:, 1] <= 64).all()

    def test_zero_new(self):
        """Requesting 0 new centers returns empty tensor."""
        rendered = torch.zeros(3, 64, 64)
        target = torch.ones(3, 64, 64)
        centers = compute_densify_centers(rendered, target, 0, 64, 64)
        assert centers.shape == (0, 2)

    def test_hotspots_at_error_regions(self):
        """Centers concentrate near high-error regions."""
        rendered = torch.zeros(3, 64, 64)
        target = torch.zeros(3, 64, 64)
        # Place a high-error patch in the top-left corner
        target[:, 0:16, 0:16] = 1.0
        centers = compute_densify_centers(rendered, target, 1, 64, 64)
        # The center should be in the top-left quadrant
        assert centers[0, 0].item() < 32  # x
        assert centers[0, 1].item() < 32  # y

    def test_nodiff_threshold(self):
        """Low errors below the threshold are zeroed out."""
        rendered = torch.zeros(3, 64, 64)
        target = torch.full((3, 64, 64), 0.01)  # very small error
        centers = compute_densify_centers(rendered, target, 5, 64, 64, nodiff_threshold=0.05)
        # All errors are below threshold, so all cells have zero error
        # Still returns centers (from zero-error cells), but they're not meaningful
        assert centers.shape[0] <= 5


# ── color_distance ───────────────────────────────────────────────────────


class TestComputeColorDistance:
    """Test pairwise color distance computation."""

    def test_same_colors_zero_distance(self):
        """Identical pre-sigmoid colors have zero distance."""
        colors = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        cdist = compute_color_distance(colors)
        assert cdist[0, 1].item() == pytest.approx(0.0, abs=1e-5)

    def test_different_colors_positive_distance(self):
        """Different colors have positive distance."""
        colors = torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])
        cdist = compute_color_distance(colors)
        assert cdist[0, 1].item() > 0.5

    def test_symmetry(self):
        """Color distance matrix is symmetric."""
        colors = torch.randn(5, 3)
        cdist = compute_color_distance(colors)
        torch.testing.assert_close(cdist, cdist.T)
