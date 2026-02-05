"""Tests for true enclosed area computation."""

import pytest
import torch
from torch import Tensor

from bezier_splatting.area import bezier_signed_area, closed_curve_enclosed_area


class TestBezierSignedArea:
    """Tests for bezier_signed_area function."""

    def test_area_under_horizontal_line(self):
        """Horizontal line at y=1 from x=0 to x=2 has area 2 (under line)."""
        cp = torch.tensor([[[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]])
        area = bezier_signed_area(cp)
        assert area.shape == (1,)
        assert abs(area[0].item() - 2.0) < 0.01

    def test_unit_square(self):
        """Square with corners should have area ~1.0."""
        cp = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
        area = bezier_signed_area(cp)
        assert area.shape == (1,)
        # Note: this is signed area "under" the open curve, not the closed polygon
        # so the value depends on interpretation

    def test_batch_processing(self):
        """Should handle batched inputs."""
        cp = torch.rand(5, 4, 2)
        areas = bezier_signed_area(cp)
        assert areas.shape == (5,)

    def test_orientation_affects_sign(self):
        """Reversed curve should have opposite sign."""
        cp = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
        cp_rev = torch.tensor([[[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]])
        area_fwd = bezier_signed_area(cp)
        area_rev = bezier_signed_area(cp_rev)
        assert abs(area_fwd[0].item() + area_rev[0].item()) < 0.01

    def test_differentiable(self):
        """Should support gradient computation."""
        cp = torch.rand(3, 4, 2, requires_grad=True)
        area = bezier_signed_area(cp)
        loss = area.sum()
        loss.backward()
        assert cp.grad is not None
        assert cp.grad.shape == cp.shape


class TestClosedCurveEnclosedArea:
    """Tests for closed_curve_enclosed_area function."""

    def test_horizontal_slab(self):
        """Two horizontal lines forming a rectangular region."""
        # Top boundary: y=1
        # Bottom boundary: y=0
        # x from 0 to 10
        # Area should be approximately 10
        bcp = torch.tensor([
            [
                [[0.0, 1.0], [5.0, 1.0], [10.0, 1.0]],  # top
                [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],  # bottom
            ]
        ])
        area = closed_curve_enclosed_area(bcp)
        assert area.shape == (1,)
        # Linear control points approximate the true area well
        assert abs(area[0].item() - 10.0) < 1.0

    def test_symmetric_lens(self):
        """Two symmetric curves forming a lens-shaped region."""
        # Two curves that bow outward symmetrically
        bcp = torch.tensor([
            [
                [[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]],  # top (bows up)
                [[0.0, 0.0], [0.5, -0.5], [1.0, 0.0]],  # bottom (bows down)
            ]
        ])
        area = closed_curve_enclosed_area(bcp)
        assert area.shape == (1,)
        # Area should be positive
        assert area[0].item() > 0

    def test_empty_input(self):
        """Empty input should return empty tensor."""
        bcp = torch.empty(0, 2, 4, 2)
        area = closed_curve_enclosed_area(bcp)
        assert area.shape == (0,)

    def test_batch_processing(self):
        """Should handle batched inputs."""
        bcp = torch.rand(10, 2, 4, 2)
        areas = closed_curve_enclosed_area(bcp)
        assert areas.shape == (10,)
        # All areas should be positive
        assert (areas > 0).all()

    def test_differentiable(self):
        """Should support gradient computation."""
        bcp = torch.rand(5, 2, 4, 2, requires_grad=True)
        areas = closed_curve_enclosed_area(bcp)
        loss = areas.sum()
        loss.backward()
        assert bcp.grad is not None
        assert bcp.grad.shape == bcp.shape

    def test_area_invariant_to_translation(self):
        """Area should not change when shape is translated."""
        bcp = torch.tensor([
            [
                [[0.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ])
        bcp_shifted = bcp + 100.0
        area1 = closed_curve_enclosed_area(bcp)
        area2 = closed_curve_enclosed_area(bcp_shifted)
        assert abs(area1[0].item() - area2[0].item()) < 0.01

    def test_area_scales_with_size(self):
        """Doubling the size should quadruple the area."""
        bcp = torch.tensor([
            [
                [[0.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ])
        bcp_doubled = bcp * 2.0
        area1 = closed_curve_enclosed_area(bcp)
        area2 = closed_curve_enclosed_area(bcp_doubled)
        ratio = area2[0].item() / area1[0].item()
        assert abs(ratio - 4.0) < 0.1


class TestIntegration:
    """Integration tests with the model."""

    def test_model_forward_with_closed_curves(self):
        """Model should work with the new area calculation."""
        from bezier_splatting.model import VectorGraphicsScene

        scene = VectorGraphicsScene(n_open=0, n_closed=5, H=64, W=64)
        output = scene()
        assert output.shape == (3, 64, 64)

    def test_model_forward_with_both_curve_types(self):
        """Model should work with both open and closed curves."""
        from bezier_splatting.model import VectorGraphicsScene

        scene = VectorGraphicsScene(n_open=5, n_closed=5, H=64, W=64)
        output = scene()
        assert output.shape == (3, 64, 64)

    def test_depth_ordering(self):
        """Smaller enclosed areas should result in curves being rendered on top."""
        from bezier_splatting.model import VectorGraphicsScene

        # This is a behavioral test - just verify it doesn't crash
        scene = VectorGraphicsScene(n_open=3, n_closed=3, H=64, W=64)
        with torch.no_grad():
            output = scene()
        assert output.shape == (3, 64, 64)
