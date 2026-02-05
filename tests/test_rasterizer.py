"""Unit tests for the tile-based Gaussian rasterizer."""

import torch
import pytest

from bezier_splatting.rasterizer import rasterize, _build_covariance, _invert_2x2
from bezier_splatting.sampling import GaussianParams


class TestCovariance:
    def test_identity_at_zero_rotation(self):
        """With θ=0, Σ = diag(σ_x², σ_y²)."""
        scales = torch.tensor([[2.0, 3.0]])
        rotations = torch.tensor([0.0])
        cov = _build_covariance(scales, rotations)

        expected = torch.tensor([[[4.0, 0.0], [0.0, 9.0]]])
        assert torch.allclose(cov, expected, atol=1e-5)

    def test_symmetric(self):
        """Covariance should be symmetric."""
        scales = torch.rand(10, 2) + 0.1
        rotations = torch.rand(10) * 2 * 3.14159
        cov = _build_covariance(scales, rotations)
        assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)

    def test_positive_definite(self):
        """Covariance should be positive definite (positive determinant)."""
        scales = torch.rand(10, 2) + 0.1
        rotations = torch.rand(10) * 2 * 3.14159
        cov = _build_covariance(scales, rotations)
        det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
        assert (det > 0).all()


class TestInvert2x2:
    def test_inverse_identity(self):
        """Inverse of scaled identity should be inverse scaling."""
        cov = torch.tensor([[[4.0, 0.0], [0.0, 9.0]]])
        inv, det = _invert_2x2(cov)
        expected_inv = torch.tensor([[[1/4, 0.0], [0.0, 1/9]]])
        assert torch.allclose(inv, expected_inv, atol=1e-5)
        assert torch.isclose(det[0], torch.tensor(36.0), atol=1e-4)

    def test_inverse_product_identity(self):
        """Σ @ Σ⁻¹ should be identity."""
        scales = torch.rand(5, 2) + 0.5
        rotations = torch.rand(5) * 2 * 3.14159
        cov = _build_covariance(scales, rotations)
        inv, _ = _invert_2x2(cov)
        product = torch.bmm(cov, inv)
        eye = torch.eye(2).expand(5, -1, -1)
        assert torch.allclose(product, eye, atol=1e-4)


class TestRasterize:
    def test_empty_produces_background(self):
        """No Gaussians → solid background color."""
        g = GaussianParams(
            means=torch.empty(0, 2),
            scales=torch.empty(0, 2),
            rotations=torch.empty(0),
            colors=torch.empty(0, 3),
            opacities=torch.empty(0),
            curve_ids=torch.empty(0, dtype=torch.long),
        )
        img = rasterize(g, 32, 32)
        assert img.shape == (3, 32, 32)
        assert torch.allclose(img, torch.ones(3, 32, 32), atol=1e-5)

    def test_single_gaussian_visible(self):
        """A single high-opacity Gaussian should produce a visible blob."""
        g = GaussianParams(
            means=torch.tensor([[16.0, 16.0]]),
            scales=torch.tensor([[3.0, 3.0]]),
            rotations=torch.tensor([0.0]),
            colors=torch.tensor([[1.0, 0.0, 0.0]]),  # red
            opacities=torch.tensor([5.0]),  # sigmoid(5) ≈ 0.99
            curve_ids=torch.tensor([0]),
        )
        img = rasterize(g, 32, 32)
        assert img.shape == (3, 32, 32)

        # Center pixel should be red-ish
        center_r = img[0, 16, 16].item()
        center_g = img[1, 16, 16].item()
        assert center_r > 0.9, f"Expected red center, got R={center_r:.3f}"
        assert center_g < 0.2, f"Expected low green, got G={center_g:.3f}"

    def test_output_range(self):
        """Output should be in [0, 1]."""
        g = GaussianParams(
            means=torch.rand(10, 2) * 64,
            scales=torch.rand(10, 2) * 3 + 1,
            rotations=torch.rand(10) * 6.28,
            colors=torch.rand(10, 3),
            opacities=torch.randn(10),
            curve_ids=torch.arange(10),
        )
        img = rasterize(g, 64, 64)
        assert img.min() >= -0.01
        assert img.max() <= 1.01

    def test_tile_boundary_continuity(self):
        """Gaussian spanning tile boundary should render smoothly."""
        # Place Gaussian right on a tile boundary (tile_size=16, so boundary at x=16)
        g = GaussianParams(
            means=torch.tensor([[16.0, 16.0]]),
            scales=torch.tensor([[5.0, 5.0]]),
            rotations=torch.tensor([0.0]),
            colors=torch.tensor([[0.0, 1.0, 0.0]]),
            opacities=torch.tensor([3.0]),
            curve_ids=torch.tensor([0]),
        )
        img = rasterize(g, 32, 32, tile_size=16)

        # Pixels at tile boundary should not show discontinuity
        # Check that pixel 15 and pixel 16 are similar
        left = img[:, 16, 15]
        right = img[:, 16, 16]
        diff = (left - right).abs().max().item()
        assert diff < 0.05, f"Tile boundary discontinuity: {diff:.4f}"

    def test_front_to_back_compositing_order(self):
        """Index 0 = frontmost. A small opaque Gaussian at index 0 should dominate."""
        # Small blue at index 0 (front), large red at index 1 (back)
        g = GaussianParams(
            means=torch.tensor([[16.0, 16.0], [16.0, 16.0]]),
            scales=torch.tensor([[2.0, 2.0], [10.0, 10.0]]),
            rotations=torch.tensor([0.0, 0.0]),
            colors=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            opacities=torch.tensor([5.0, 5.0]),  # both near-opaque
            curve_ids=torch.tensor([0, 1]),
        )
        img = rasterize(g, 32, 32)
        center = img[:, 16, 16]
        # Blue (index 0, front) should dominate over red (index 1, back)
        assert center[2] > center[0], (
            f"Front-to-back order broken: blue={center[2]:.3f} should > red={center[0]:.3f}"
        )
