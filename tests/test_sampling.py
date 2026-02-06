"""Unit tests for Gaussian sampling from curves.

Control points are in [-1, 1] normalized coordinates.
Samplers scale to pixel coords internally via H, W args.
"""

import torch
import pytest

from bezier_splatting.sampling import OpenCurveSampler, ClosedCurveSampler, GaussianParams


class TestOpenCurveSampler:
    def test_output_count(self):
        """Should produce samples_per_curve * num_curves Gaussians."""
        sampler = OpenCurveSampler(samples_per_curve=20)
        cp = torch.rand(3, 10, 2) * 2 - 1  # [-1,1] normalized
        colors = torch.rand(3, 3)
        opacities = torch.zeros(3, 3)  # per-segment
        stroke_widths = torch.zeros(3)

        g = sampler(cp, colors, opacities, stroke_widths, H=256, W=256)
        assert g.means.shape[0] == 60  # 3 curves * 20 samples

    def test_output_shapes(self):
        """All output tensors should have consistent shapes."""
        sampler = OpenCurveSampler(samples_per_curve=15)
        cp = torch.rand(2, 10, 2) * 2 - 1  # [-1,1] normalized
        colors = torch.rand(2, 3)
        opacities = torch.zeros(2, 3)  # per-segment
        stroke_widths = torch.zeros(2)

        g = sampler(cp, colors, opacities, stroke_widths, H=256, W=256)
        G = 30  # 2 * 15
        assert g.means.shape == (G, 2)
        assert g.scales.shape == (G, 2)
        assert g.rotations.shape == (G,)
        assert g.colors.shape == (G, 3)
        assert g.opacities.shape == (G,)
        assert g.curve_ids.shape == (G,)

    def test_positive_scales(self):
        """Scales should be positive."""
        sampler = OpenCurveSampler(samples_per_curve=20)
        cp = torch.rand(5, 10, 2) * 2 - 1
        g = sampler(cp, torch.rand(5, 3), torch.zeros(5, 3), torch.zeros(5), H=256, W=256)
        assert (g.scales > 0).all()

    def test_empty_curves(self):
        """Should handle zero curves gracefully."""
        sampler = OpenCurveSampler()
        cp = torch.empty(0, 10, 2)
        g = sampler(cp, torch.empty(0, 3), torch.empty(0, 3), torch.empty(0), H=256, W=256)
        assert g.means.shape[0] == 0

    def test_curve_ids(self):
        """Curve IDs should correctly identify which curve each Gaussian belongs to."""
        sampler = OpenCurveSampler(samples_per_curve=10)
        cp = torch.rand(3, 10, 2) * 2 - 1
        g = sampler(cp, torch.rand(3, 3), torch.zeros(3, 3), torch.zeros(3), H=256, W=256)

        # Each curve should have 10 Gaussians with the same ID
        for i in range(3):
            curve_mask = g.curve_ids == i
            assert curve_mask.sum() == 10, f"Curve {i} should have 10 Gaussians"

    def test_curve_id_offset(self):
        """Curve IDs should be offset by curve_id_offset."""
        sampler = OpenCurveSampler(samples_per_curve=10)
        cp = torch.rand(2, 10, 2) * 2 - 1
        g = sampler(cp, torch.rand(2, 3), torch.zeros(2, 3), torch.zeros(2),
                     H=256, W=256, curve_id_offset=5)
        assert g.curve_ids.min() == 5
        assert g.curve_ids.max() == 6

    @pytest.mark.parametrize("K", [9, 10, 11, 19, 20, 21])
    def test_segment_opacity_alignment(self, K):
        """Per-segment opacity bins must match evaluate_composite_bezier's sample distribution."""
        sampler = OpenCurveSampler(samples_per_curve=K)
        N = 1
        cp = torch.rand(N, 10, 2) * 2 - 1
        # Use distinct opacity values per segment so we can detect misalignment
        opacities = torch.tensor([[1.0, 2.0, 3.0]])
        g = sampler(cp, torch.rand(N, 3), opacities, torch.zeros(N), H=256, W=256)

        # Reconstruct what evaluate_composite_bezier does
        base = K // 3
        remainder = K - 3 * base
        expected_sizes = [base + (1 if i < remainder else 0) for i in range(3)]

        # Verify each segment's opacities are correct
        k = 0
        for seg, n_seg in enumerate(expected_sizes):
            expected_val = opacities[0, seg].item()
            actual = g.opacities[k:k + n_seg]
            assert (actual == expected_val).all(), (
                f"K={K}, seg={seg}: expected opacity {expected_val} for "
                f"samples [{k}:{k + n_seg}], got {actual.tolist()}"
            )
            k += n_seg

    def test_means_in_pixel_space(self):
        """Output means should be in pixel coordinates (scaled from [-1,1])."""
        sampler = OpenCurveSampler(samples_per_curve=10)
        cp = torch.rand(2, 10, 2) * 2 - 1  # [-1,1]
        g = sampler(cp, torch.rand(2, 3), torch.zeros(2, 3), torch.zeros(2), H=128, W=256)
        # Means should be in pixel space [0, W] × [0, H]
        assert g.means[:, 0].max() <= 256 + 1
        assert g.means[:, 1].max() <= 128 + 1


class TestClosedCurveSampler:
    def test_output_count(self):
        """Should produce (R+2) * samples_per_curve * num_curves Gaussians.

        R+2 = num_intermediate + 2 boundaries (paper Eq. 6).
        """
        R = 8
        K = 10
        sampler = ClosedCurveSampler(num_intermediate=R, samples_per_curve=K)
        bcp = torch.rand(2, 2, 4, 2) * 2 - 1  # [-1,1] normalized
        colors = torch.rand(2, 3)
        opacities = torch.zeros(2)

        g = sampler(bcp, colors, opacities, H=256, W=256)
        R_total = R + 2
        expected = 2 * R_total * K  # 2 curves * 10 total rows * 10 samples
        assert g.means.shape[0] == expected, (
            f"Expected {expected} Gaussians, got {g.means.shape[0]}"
        )

    def test_output_shapes(self):
        R = 4
        K = 5
        sampler = ClosedCurveSampler(num_intermediate=R, samples_per_curve=K)
        bcp = torch.rand(3, 2, 4, 2) * 2 - 1
        g = sampler(bcp, torch.rand(3, 3), torch.zeros(3), H=256, W=256)
        R_total = R + 2
        G = 3 * R_total * K
        assert g.means.shape == (G, 2)
        assert g.scales.shape == (G, 2)
        assert g.rotations.shape == (G,)
        assert g.curve_ids.shape == (G,)

    def test_empty_curves(self):
        sampler = ClosedCurveSampler()
        bcp = torch.empty(0, 2, 4, 2)
        g = sampler(bcp, torch.empty(0, 3), torch.empty(0), H=256, W=256)
        assert g.means.shape[0] == 0

    def test_curve_ids(self):
        """All Gaussians from the same closed curve should share a curve ID."""
        R = 4
        K = 5
        sampler = ClosedCurveSampler(num_intermediate=R, samples_per_curve=K)
        bcp = torch.rand(2, 2, 4, 2) * 2 - 1
        g = sampler(bcp, torch.rand(2, 3), torch.zeros(2), H=256, W=256)

        R_total = R + 2
        expected_per_curve = R_total * K
        for i in range(2):
            curve_mask = g.curve_ids == i
            assert curve_mask.sum() == expected_per_curve


class TestGaussianParams:
    def test_concat(self):
        """Concatenation should combine two GaussianParams."""
        g1 = GaussianParams(
            means=torch.rand(5, 2),
            scales=torch.rand(5, 2),
            rotations=torch.rand(5),
            colors=torch.rand(5, 3),
            opacities=torch.rand(5),
            curve_ids=torch.arange(5),
        )
        g2 = GaussianParams(
            means=torch.rand(3, 2),
            scales=torch.rand(3, 2),
            rotations=torch.rand(3),
            colors=torch.rand(3, 3),
            opacities=torch.rand(3),
            curve_ids=torch.arange(3) + 5,
        )
        combined = g1.concat(g2)
        assert combined.means.shape[0] == 8
        assert combined.colors.shape == (8, 3)
        assert combined.curve_ids.shape == (8,)
