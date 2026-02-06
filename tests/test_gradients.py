"""Gradient sanity tests — verify autograd flows through the full pipeline."""

import torch
import torch.nn.functional as F
import pytest

from bezier_splatting.model import VectorGraphicsScene
from bezier_splatting.rasterizer import rasterize
from bezier_splatting.sampling import GaussianParams


class TestGradientFlow:
    def test_gradient_flows_to_control_points(self):
        """Render → MSE loss → backward. Assert open_control_points.grad is non-zero."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32, samples_per_open=5)
        target = torch.rand(3, 32, 32)

        rendered = scene(32, 32)
        loss = F.mse_loss(rendered, target)
        loss.backward()

        assert scene.open_control_points.grad is not None
        assert scene.open_control_points.grad.abs().max() > 0, "No gradient on control points"

    def test_gradient_flows_to_colors(self):
        """Gradient should flow to color parameters."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32, samples_per_open=5)
        target = torch.rand(3, 32, 32)

        rendered = scene(32, 32)
        loss = F.mse_loss(rendered, target)
        loss.backward()

        assert scene.open_colors.grad is not None
        assert scene.open_colors.grad.abs().max() > 0, "No gradient on colors"

    def test_gradient_flows_to_opacity(self):
        """Gradient should flow to opacity. Sigmoid must not saturate."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32, samples_per_open=5)
        # Initialize opacities near 0 (sigmoid ≈ 0.5) to avoid saturation
        # Shape is (4, 3) — per-segment opacity
        scene.open_opacities = torch.nn.Parameter(torch.zeros(4, 3))
        target = torch.rand(3, 32, 32)

        rendered = scene(32, 32)
        loss = F.mse_loss(rendered, target)
        loss.backward()

        assert scene.open_opacities.grad is not None
        assert scene.open_opacities.grad.abs().max() > 0, "No gradient on opacities"

    def test_gradient_flows_to_stroke_width(self):
        """Gradient should flow to stroke width."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32, samples_per_open=5)
        target = torch.rand(3, 32, 32)

        rendered = scene(32, 32)
        loss = F.mse_loss(rendered, target)
        loss.backward()

        assert scene.open_stroke_widths.grad is not None
        assert scene.open_stroke_widths.grad.abs().max() > 0, "No gradient on stroke widths"

    def test_gradient_flows_to_closed_curves(self):
        """Gradient should flow to closed curve boundary CPs."""
        scene = VectorGraphicsScene(n_open=0, n_closed=4, H=32, W=32, samples_per_closed_curve=5, num_intermediate=4)
        target = torch.rand(3, 32, 32)

        rendered = scene(32, 32)
        loss = F.mse_loss(rendered, target)
        loss.backward()

        assert scene.closed_shared_pts.grad is not None
        assert scene.closed_shared_pts.grad.abs().max() > 0, "No gradient on shared endpoints"
        assert scene.closed_interior_cp.grad is not None
        assert scene.closed_interior_cp.grad.abs().max() > 0, "No gradient on interior CPs"


class TestRasterizerDifferentiable:
    def test_finite_difference_single_gaussian(self):
        """Compare autograd gradient with numerical gradient for a single Gaussian."""
        means = torch.tensor([[16.0, 16.0]], requires_grad=True)
        scales = torch.tensor([[3.0, 3.0]])
        rotations = torch.tensor([0.0])
        colors = torch.tensor([[1.0, 0.0, 0.0]])
        opacities = torch.tensor([2.0])
        curve_ids = torch.tensor([0])

        target = torch.rand(3, 32, 32)

        # Autograd gradient
        g = GaussianParams(means=means, scales=scales, rotations=rotations,
                           colors=colors, opacities=opacities, curve_ids=curve_ids)
        img = rasterize(g, 32, 32)
        loss = F.mse_loss(img, target)
        loss.backward()
        autograd_grad = means.grad.clone()

        # Numerical gradient (finite differences)
        eps = 1e-3
        numerical_grad = torch.zeros_like(means)
        for i in range(2):
            means_plus = means.detach().clone()
            means_plus[0, i] += eps
            g_plus = GaussianParams(means=means_plus, scales=scales, rotations=rotations,
                                    colors=colors, opacities=opacities, curve_ids=curve_ids)
            loss_plus = F.mse_loss(rasterize(g_plus, 32, 32), target)

            means_minus = means.detach().clone()
            means_minus[0, i] -= eps
            g_minus = GaussianParams(means=means_minus, scales=scales, rotations=rotations,
                                     colors=colors, opacities=opacities, curve_ids=curve_ids)
            loss_minus = F.mse_loss(rasterize(g_minus, 32, 32), target)

            numerical_grad[0, i] = (loss_plus - loss_minus) / (2 * eps)

        # Should agree within reasonable tolerance
        assert torch.allclose(autograd_grad, numerical_grad, atol=1e-2, rtol=0.1), (
            f"Autograd: {autograd_grad}\nNumerical: {numerical_grad}"
        )
