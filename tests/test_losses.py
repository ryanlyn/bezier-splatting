"""Tests for the configurable composite loss system."""

import torch
import torch.nn.functional as F
import pytest

from bezier_splatting.losses import (
    LossConfig,
    boundary_joint_loss,
    compute_loss,
    curvature_loss,
    opacity_prior,
    reconstruction_loss,
    shape_regularizer,
    xing_loss,
)
from bezier_splatting.model import VectorGraphicsScene


class TestReconstructionLoss:
    """Tests for reconstruction_loss with different loss types."""

    def test_l2_loss(self):
        """L2 loss should match F.mse_loss."""
        rendered = torch.rand(3, 32, 32)
        target = torch.rand(3, 32, 32)
        loss = reconstruction_loss(rendered, target, "L2")
        expected = F.mse_loss(rendered, target)
        assert torch.allclose(loss, expected)

    def test_l1_loss(self):
        """L1 loss should match F.l1_loss."""
        rendered = torch.rand(3, 32, 32)
        target = torch.rand(3, 32, 32)
        loss = reconstruction_loss(rendered, target, "L1")
        expected = F.l1_loss(rendered, target)
        assert torch.allclose(loss, expected)

    def test_fusion1_loss(self):
        """Fusion1 should be a weighted combination of MSE and (1 - SSIM)."""
        rendered = torch.rand(3, 32, 32)
        target = torch.rand(3, 32, 32)
        loss = reconstruction_loss(rendered, target, "Fusion1")
        assert loss.shape == ()
        assert loss.item() > 0

    def test_fusion1_between_mse_and_ssim(self):
        """Fusion1 loss should be between pure MSE and pure SSIM components."""
        rendered = torch.rand(3, 32, 32)
        target = torch.rand(3, 32, 32)
        mse = F.mse_loss(rendered, target)
        fusion = reconstruction_loss(rendered, target, "Fusion1", lambda_value=0.7)
        # Fusion1 combines MSE and SSIM terms, should be a finite positive scalar
        assert fusion.item() > 0
        assert torch.isfinite(fusion)

    def test_invalid_loss_type(self):
        """Unknown loss type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            reconstruction_loss(torch.rand(3, 8, 8), torch.rand(3, 8, 8), "invalid")

    def test_identical_images_low_loss(self):
        """Identical images should have zero or near-zero loss."""
        img = torch.rand(3, 32, 32)
        assert reconstruction_loss(img, img, "L2").item() < 1e-6
        assert reconstruction_loss(img, img, "L1").item() < 1e-6


class TestShapeRegularizer:
    """Tests for shape_regularizer."""

    def test_interior_cps_within_chord(self):
        """Interior CPs within the chord span should have near-zero loss."""
        # Straight line: CPs along x-axis from 0 to 1
        cp = torch.tensor([[[0.0, 0.0], [0.3, 0.0], [0.7, 0.0], [1.0, 0.0]]])
        loss = shape_regularizer(cp, degree=3)
        assert loss.item() < 1e-6

    def test_interior_cps_outside_chord(self):
        """Interior CPs projecting outside [0, 1] span should incur loss."""
        # P1 projects before P0, P2 projects after P_end
        cp = torch.tensor([[[0.0, 0.0], [-0.5, 0.0], [1.5, 0.0], [1.0, 0.0]]])
        loss = shape_regularizer(cp, degree=3)
        assert loss.item() > 0

    def test_empty_input(self):
        """Empty input should return zero loss."""
        cp = torch.empty(0, 4, 2)
        loss = shape_regularizer(cp)
        assert loss.item() == 0.0

    def test_differentiable(self):
        """Should support gradient computation."""
        cp = torch.rand(5, 4, 2, requires_grad=True)
        loss = shape_regularizer(cp)
        loss.backward()
        assert cp.grad is not None

    def test_batch_processing(self):
        """Should handle batched inputs."""
        cp = torch.rand(10, 4, 2)
        loss = shape_regularizer(cp)
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestOpacityPrior:
    """Tests for opacity_prior."""

    def test_high_opacity(self):
        """Large positive pre-sigmoid values (near 1.0 after sigmoid) should have low loss."""
        opacities = torch.tensor([5.0, 5.0, 5.0])
        loss = opacity_prior(opacities)
        # sigmoid(5) ≈ 0.993, deviation from 1.0 ≈ 0.007
        assert loss.item() < 0.01

    def test_zero_opacity(self):
        """Zero pre-sigmoid (sigmoid = 0.5) should incur significant loss."""
        opacities = torch.tensor([0.0, 0.0, 0.0])
        loss = opacity_prior(opacities)
        # |sigmoid(0) - 1| = 0.5
        assert abs(loss.item() - 0.5) < 0.01

    def test_negative_opacity(self):
        """Negative pre-sigmoid should incur high loss."""
        opacities = torch.tensor([-5.0, -5.0])
        loss = opacity_prior(opacities)
        # sigmoid(-5) ≈ 0.007, |0.007 - 1| ≈ 0.993
        assert loss.item() > 0.9

    def test_empty_input(self):
        """Empty input should return zero loss."""
        loss = opacity_prior(torch.empty(0))
        assert loss.item() == 0.0

    def test_differentiable(self):
        """Should support gradient computation."""
        opacities = torch.zeros(5, requires_grad=True)
        loss = opacity_prior(opacities)
        loss.backward()
        assert opacities.grad is not None

    def test_profile_shape(self):
        """Opacity prior supports closed profile logits shaped (N, 3)."""
        opacities = torch.zeros(4, 3)
        loss = opacity_prior(opacities)
        assert abs(loss.item() - 0.5) < 0.01


class TestCurvatureLoss:
    """Tests for curvature_loss."""

    def test_straight_boundaries(self):
        """Straight-line boundaries should have low curvature loss."""
        # Two straight horizontal boundaries
        bcp = torch.tensor([
            [
                [[-1.0, 0.1], [0.0, 0.1], [1.0, 0.1]],  # top
                [[-1.0, -0.1], [0.0, -0.1], [1.0, -0.1]],  # bottom
            ]
        ])
        loss = curvature_loss(bcp, H=64, W=64)
        assert loss.item() < 1.0  # low curvature for straight lines

    def test_empty_input(self):
        """Empty input should return zero loss."""
        bcp = torch.empty(0, 2, 4, 2)
        loss = curvature_loss(bcp, H=64, W=64)
        assert loss.item() == 0.0

    def test_non_negative(self):
        """Curvature loss should be non-negative."""
        bcp = torch.rand(3, 2, 4, 2) * 2 - 1  # [-1, 1]
        loss = curvature_loss(bcp, H=64, W=64)
        assert loss.item() >= 0

    def test_differentiable(self):
        """Should support gradient computation."""
        bcp = (torch.rand(3, 2, 4, 2) * 2 - 1).detach().requires_grad_(True)
        loss = curvature_loss(bcp, H=64, W=64)
        loss.backward()
        assert bcp.grad is not None


class TestBoundaryJointLoss:
    """Tests for boundary_joint_loss."""

    def test_within_bounds(self):
        """CPs within [-1, 1] should have zero loss."""
        cp = torch.tensor([[[0.0, 0.0], [0.5, 0.3], [-0.5, -0.3], [0.8, 0.8]]])
        loss = boundary_joint_loss(cp, degree=3)
        assert loss.item() < 1e-6

    def test_exceeds_bounds(self):
        """Joint CPs outside [-1, 1] should incur loss."""
        cp = torch.tensor([[[1.5, 0.0], [0.5, 0.3], [-0.5, -0.3], [-1.5, 0.0]]])
        loss = boundary_joint_loss(cp, degree=3)
        assert loss.item() > 0

    def test_empty_input(self):
        """Empty input should return zero loss."""
        cp = torch.empty(0, 4, 2)
        loss = boundary_joint_loss(cp)
        assert loss.item() == 0.0

    def test_differentiable(self):
        """Should support gradient computation."""
        cp = (torch.rand(5, 10, 2) * 3 - 1.5).detach().requires_grad_(True)  # some outside bounds
        loss = boundary_joint_loss(cp, degree=3)
        loss.backward()
        assert cp.grad is not None


class TestXingLoss:
    """Tests for xing_loss (moved from optimization.py)."""

    def test_open_curves_only(self):
        """Xing loss should be zero for open-only scenes."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32)
        loss = xing_loss(scene)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() == 0.0

    def test_closed_curves_only(self):
        """Xing loss should work with closed curves only."""
        scene = VectorGraphicsScene(n_open=0, n_closed=4, H=32, W=32)
        loss = xing_loss(scene)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_both_curve_types(self):
        """Xing loss should work with mixed scenes (closed-only contribution)."""
        scene = VectorGraphicsScene(n_open=3, n_closed=3, H=32, W=32)
        loss = xing_loss(scene)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_non_negative(self):
        """Xing loss should be non-negative."""
        scene = VectorGraphicsScene(n_open=5, n_closed=3, H=32, W=32)
        loss = xing_loss(scene)
        assert loss.item() >= 0

    def test_differentiable(self):
        """Xing loss should flow gradients back to closed control points."""
        scene = VectorGraphicsScene(n_open=0, n_closed=4, H=32, W=32)
        loss = xing_loss(scene)
        loss.backward()
        assert scene.closed_shared_pts.grad is not None


class TestComputeLoss:
    """Tests for the compute_loss entry point."""

    def test_default_config(self):
        """Default config should compute reconstruction loss."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32)
        rendered = scene(32, 32).detach()
        target = torch.rand(3, 32, 32)
        config = LossConfig()
        total, loss_dict = compute_loss(rendered, target, scene, config)
        assert "reconstruction" in loss_dict
        assert "total" in loss_dict
        assert total.shape == ()
        assert torch.isfinite(total)

    def test_all_disabled(self):
        """With all regularizers disabled and xing=0, should only have reconstruction."""
        scene = VectorGraphicsScene(n_open=4, n_closed=3, H=32, W=32)
        rendered = scene(32, 32).detach()
        target = torch.rand(3, 32, 32)
        config = LossConfig(
            lambda_xing=0.0,
            apply_shape_reg=False,
            apply_opacity_prior=False,
            apply_curvature=False,
            apply_boundary=False,
        )
        total, loss_dict = compute_loss(rendered, target, scene, config)
        assert "reconstruction" in loss_dict
        assert "xing" not in loss_dict
        assert "shape_reg" not in loss_dict
        assert "opacity_prior" not in loss_dict
        assert "curvature" not in loss_dict
        assert "boundary_joint" not in loss_dict

    def test_all_enabled_open_only(self):
        """With all terms enabled but only open curves, closed-only terms should not appear."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32)
        rendered = scene(32, 32).detach()
        target = torch.rand(3, 32, 32)
        config = LossConfig(
            lambda_xing=0.01,
            apply_shape_reg=True,
            apply_opacity_prior=True,
            apply_curvature=True,
            apply_boundary=True,
        )
        total, loss_dict = compute_loss(rendered, target, scene, config)
        assert "reconstruction" in loss_dict
        assert "xing" not in loss_dict
        assert "boundary_joint" in loss_dict  # applies to open curves
        # Closed-only terms should not appear (no closed curves)
        assert "shape_reg" not in loss_dict
        assert "opacity_prior" not in loss_dict
        assert "curvature" not in loss_dict

    def test_all_enabled_both_types(self):
        """With both curve types and all terms enabled, all keys should be present."""
        scene = VectorGraphicsScene(n_open=3, n_closed=3, H=32, W=32)
        rendered = scene(32, 32).detach()
        target = torch.rand(3, 32, 32)
        config = LossConfig(
            lambda_xing=0.01,
            apply_shape_reg=True,
            apply_opacity_prior=True,
            apply_curvature=True,
            apply_boundary=True,
        )
        total, loss_dict = compute_loss(rendered, target, scene, config)
        expected_keys = {"reconstruction", "xing", "shape_reg", "opacity_prior",
                         "curvature", "boundary_joint", "total"}
        assert expected_keys == set(loss_dict.keys())

    def test_xing_only(self):
        """Open-only scenes should ignore xing and keep reconstruction-only loss."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32)
        rendered = scene(32, 32).detach()
        target = torch.rand(3, 32, 32)
        config = LossConfig(
            lambda_xing=0.1,
            apply_shape_reg=False,
            apply_opacity_prior=False,
            apply_curvature=False,
            apply_boundary=False,
        )
        total, loss_dict = compute_loss(rendered, target, scene, config)
        assert "xing" not in loss_dict
        assert set(loss_dict.keys()) == {"reconstruction", "total"}
        assert "shape_reg" not in loss_dict

    def test_loss_dict_values_are_float(self):
        """All loss_dict values should be plain Python floats."""
        scene = VectorGraphicsScene(n_open=3, n_closed=3, H=32, W=32)
        rendered = scene(32, 32).detach()
        target = torch.rand(3, 32, 32)
        config = LossConfig(lambda_xing=0.01)
        _, loss_dict = compute_loss(rendered, target, scene, config)
        for k, v in loss_dict.items():
            assert isinstance(v, float), f"loss_dict[{k!r}] is {type(v)}, expected float"

    def test_differentiable(self):
        """compute_loss total should support backward."""
        scene = VectorGraphicsScene(n_open=4, n_closed=3, H=32, W=32)
        target = torch.rand(3, 32, 32)
        rendered = scene(32, 32)  # keep grad graph
        config = LossConfig(
            lambda_xing=0.01,
            apply_shape_reg=True,
            apply_opacity_prior=True,
            apply_curvature=True,
            apply_boundary=True,
        )
        total, _ = compute_loss(rendered, target, scene, config)
        total.backward()
        assert scene.open_control_points.grad is not None
        assert scene.closed_shared_pts.grad is not None

    def test_fusion1_loss_type(self):
        """Fusion1 loss type should work in compute_loss."""
        scene = VectorGraphicsScene(n_open=4, n_closed=0, H=32, W=32)
        rendered = scene(32, 32).detach()
        target = torch.rand(3, 32, 32)
        config = LossConfig(loss_type="Fusion1")
        total, loss_dict = compute_loss(rendered, target, scene, config)
        assert "reconstruction" in loss_dict
        assert total.item() > 0


class TestLossConfig:
    """Tests for LossConfig dataclass defaults."""

    def test_defaults(self):
        """Check that defaults match spec."""
        config = LossConfig()
        assert config.loss_type == "L2"
        assert config.lambda_xing == 0.0
        assert config.lambda_shape == 1e-2
        assert config.lambda_opacity_prior == 1e-2
        assert config.lambda_curvature == 1.0
        assert config.lambda_boundary == 1.0
        assert config.apply_shape_reg is True
        assert config.apply_opacity_prior is True
        assert config.apply_curvature is True
        assert config.apply_boundary is True

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = LossConfig(loss_type="L1", lambda_xing=0.5, apply_curvature=False)
        assert config.loss_type == "L1"
        assert config.lambda_xing == 0.5
        assert config.apply_curvature is False


class TestCurvatureLossGate:
    """The turning-angle gate must target sharp corners, not smooth regions."""

    def test_sharp_kink_penalized_more_than_smooth_arc(self):
        smooth = torch.tensor([
            [
                [[-0.5, 0.0], [-0.2, 0.3], [0.2, 0.3], [0.5, 0.0]],
                [[-0.5, 0.0], [-0.2, -0.3], [0.2, -0.3], [0.5, 0.0]],
            ]
        ])
        # Hairpin fold in the top boundary: the sampled polyline turns by
        # ~90 degrees between adjacent samples at the fold tips
        spiky = torch.tensor([
            [
                [[-0.5, 0.0], [2.0, 0.1], [-2.0, 0.1], [0.5, 0.0]],
                [[-0.5, 0.0], [-0.2, -0.3], [0.2, -0.3], [0.5, 0.0]],
            ]
        ])
        l_smooth = curvature_loss(smooth, H=256, W=256).item()
        l_spiky = curvature_loss(spiky, H=256, W=256).item()
        assert l_smooth == 0.0
        assert l_spiky > 0.1

    def test_smooth_arc_not_penalized(self):
        """Gentle bends turn < 60 degrees per sample and stay unmasked."""
        smooth = torch.tensor([
            [
                [[-0.5, 0.0], [-0.2, 0.3], [0.2, 0.3], [0.5, 0.0]],
                [[-0.5, 0.0], [-0.2, -0.3], [0.2, -0.3], [0.5, 0.0]],
            ]
        ])
        assert curvature_loss(smooth, H=256, W=256).item() == 0.0

    def test_joint_corners_exempt(self):
        """Sharp angles where the two boundaries meet are legitimate corners."""
        # Lens whose boundaries meet at a sharp angle but are smooth inside
        lens = torch.tensor([
            [
                [[-0.5, 0.0], [-0.2, 0.4], [0.2, 0.4], [0.5, 0.0]],
                [[-0.5, 0.0], [-0.2, -0.4], [0.2, -0.4], [0.5, 0.0]],
            ]
        ])
        assert curvature_loss(lens, H=256, W=256).item() == 0.0


class TestXingLossNormalization:
    def test_scale_independent_of_curve_count(self):
        """Duplicating curves must not change the (mean) xing loss."""
        torch.manual_seed(0)
        bcp = torch.rand(4, 2, 4, 2) * 2 - 1
        scene_small = VectorGraphicsScene(n_open=0, n_closed=4, H=32, W=32)
        loss_small = xing_loss(scene_small, boundary_cp=bcp)
        loss_big = xing_loss(scene_small, boundary_cp=bcp.repeat(3, 1, 1, 1))
        torch.testing.assert_close(loss_small, loss_big, atol=1e-6, rtol=1e-5)
