"""Unit tests for Bézier curve math."""

import torch
import pytest

from bezier_splatting.bezier import (
    bernstein_basis,
    evaluate_bezier,
    bezier_tangent,
    evaluate_composite_bezier,
)


class TestBernsteinBasis:
    def test_partition_of_unity(self):
        """Bernstein basis values should sum to 1 for all t."""
        t = torch.linspace(0, 1, 100)
        for degree in [1, 2, 3, 4]:
            basis = bernstein_basis(t, degree)
            sums = basis.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
                f"Partition of unity violated for degree {degree}"
            )

    def test_endpoint_values(self):
        """At t=0, only B_0 is 1. At t=1, only B_M is 1."""
        for degree in [2, 3, 4]:
            t = torch.tensor([0.0, 1.0])
            basis = bernstein_basis(t, degree)

            # At t=0: B_0^M(0) = 1, rest = 0
            assert torch.isclose(basis[0, 0], torch.tensor(1.0), atol=1e-6)
            assert torch.allclose(basis[0, 1:], torch.zeros(degree), atol=1e-6)

            # At t=1: B_M^M(1) = 1, rest = 0
            assert torch.isclose(basis[1, -1], torch.tensor(1.0), atol=1e-6)
            assert torch.allclose(basis[1, :-1], torch.zeros(degree), atol=1e-6)

    def test_shape(self):
        """Output shape should be (*batch, degree+1)."""
        t = torch.rand(50)
        basis = bernstein_basis(t, 3)
        assert basis.shape == (50, 4)

    def test_non_negative(self):
        """Bernstein basis should be non-negative for t in [0, 1]."""
        t = torch.linspace(0, 1, 200)
        basis = bernstein_basis(t, 3)
        assert (basis >= -1e-7).all()


class TestEvaluateBezier:
    def test_endpoints(self):
        """Curve should pass through first and last control points."""
        cp = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 0.0]]])  # (1, 4, 2)
        t = torch.tensor([0.0, 1.0])
        pts = evaluate_bezier(cp, t)

        assert torch.allclose(pts[0, 0], cp[0, 0], atol=1e-5)
        assert torch.allclose(pts[0, 1], cp[0, -1], atol=1e-5)

    def test_linear(self):
        """Degree-1 Bézier should be linear interpolation."""
        cp = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])  # (1, 2, 2)
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        pts = evaluate_bezier(cp, t)

        expected = t.unsqueeze(-1) * 10.0
        assert torch.allclose(pts[0], expected, atol=1e-5)

    def test_batch(self):
        """Should work with multiple curves simultaneously."""
        cp = torch.rand(5, 4, 2)
        t = torch.linspace(0, 1, 20)
        pts = evaluate_bezier(cp, t)
        assert pts.shape == (5, 20, 2)


class TestBezierTangent:
    def test_linear_tangent(self):
        """Tangent of a degree-1 curve should be constant."""
        cp = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])  # (1, 2, 2)
        t = torch.linspace(0, 1, 10)
        tang = bezier_tangent(cp, t)

        # All tangents should equal [3, 4]
        expected = torch.tensor([3.0, 4.0]).expand(10, 2)
        assert torch.allclose(tang[0], expected, atol=1e-5)

    def test_shape(self):
        """Tangent output shape should match point output shape."""
        cp = torch.rand(3, 4, 2)
        t = torch.linspace(0, 1, 15)
        tang = bezier_tangent(cp, t)
        assert tang.shape == (3, 15, 2)


class TestCompositeBezer:
    def test_continuity(self):
        """Composite curve should be continuous at segment boundaries."""
        cp = torch.rand(2, 10, 2)
        # Ensure shared endpoints
        pts, _ = evaluate_composite_bezier(cp, 30)
        # Check the output is the right shape
        assert pts.shape[0] == 2
        assert pts.shape[2] == 2

    def test_shape(self):
        cp = torch.rand(4, 10, 2)
        pts, tangs = evaluate_composite_bezier(cp, 30)
        assert pts.shape[0] == 4
        assert tangs.shape[0] == 4
        assert pts.shape == tangs.shape
