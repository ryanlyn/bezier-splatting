"""Tests for VectorGraphicsScene construction and checkpoint compatibility."""

import torch

from bezier_splatting.model import VectorGraphicsScene


class TestClosedOpacityCompatibility:
    def test_closed_opacity_profile_shape(self):
        scene = VectorGraphicsScene(n_open=0, n_closed=3, H=32, W=32)
        assert scene.closed_opacities.shape == (3, 3)

    def test_load_legacy_scalar_closed_opacity_checkpoint(self):
        scene = VectorGraphicsScene(n_open=1, n_closed=2, H=32, W=32)
        sd = scene.state_dict()
        sd["closed_opacities"] = torch.tensor([0.25, -0.5])  # legacy shape (N,)

        scene.load_state_dict(sd, strict=True)
        assert scene.closed_opacities.shape == (2, 3)
        expected = torch.tensor([[0.25, 0.25, 0.25], [-0.5, -0.5, -0.5]])
        torch.testing.assert_close(scene.closed_opacities.detach(), expected)


class TestClosedSamplingMode:
    def test_scene_forwards_closed_sampling_mode(self):
        scene = VectorGraphicsScene(
            n_open=0,
            n_closed=1,
            H=32,
            W=32,
            closed_sampling_mode="cdf",
        )
        assert scene.closed_sampler.sampling_mode == "cdf"


class TestDepthHeuristicScale:
    """Depth heuristics must share one comparable, non-saturating scale."""

    def test_open_depths_distinct_and_fractional(self):
        torch.manual_seed(0)
        scene = VectorGraphicsScene(n_open=8, n_closed=4, H=256, W=256)
        scene.update_depth_heuristic(256, 256, update_open=True, update_closed=True)
        depths = scene.get_depth.squeeze(-1)
        open_d = depths[:8]
        closed_d = depths[8:]
        # No sigmoid-style saturation collapse: strokes keep distinct ordering
        assert open_d.unique().numel() == 8
        # Both curve types express depths as image-coverage fractions
        assert (open_d > 0).all() and (open_d < 1.0).all()
        assert (closed_d > 0).all() and (closed_d < 1.0).all()

    def test_small_stroke_sorts_in_front_of_large_fill(self):
        scene = VectorGraphicsScene(n_open=1, n_closed=1, H=64, W=64)
        with torch.no_grad():
            # Tiny open stroke
            cp = torch.linspace(-0.02, 0.02, 10).unsqueeze(-1).expand(-1, 2).clone()
            scene.open_control_points[0] = cp
            # Huge closed fill
            scene.closed_shared_pts[0] = torch.tensor([[-0.9, 0.0], [0.9, 0.0]])
            scene.closed_interior_cp[0, 0] = torch.tensor([[-0.5, 0.9], [0.5, 0.9]])
            scene.closed_interior_cp[0, 1] = torch.tensor([[-0.5, -0.9], [0.5, -0.9]])
        scene.update_depth_heuristic(64, 64, update_open=True, update_closed=True)
        depths = scene.get_depth.squeeze(-1)
        assert depths[0] < depths[1]  # stroke in front of fill


class TestOpenColorGradients:
    """Out-of-range open colors must keep receiving gradients."""

    def test_out_of_range_color_gets_gradient(self):
        scene = VectorGraphicsScene(n_open=2, n_closed=0, H=32, W=32)
        with torch.no_grad():
            scene.open_colors.fill_(1.5)  # push outside [0, 1]
            scene.open_opacities.fill_(3.0)  # make strokes clearly visible
        rendered = scene(32, 32)
        loss = rendered.mean()
        loss.backward()
        grad = scene.open_colors.grad
        assert grad is not None
        # Straight-through clamp: gradient is non-zero despite clamped forward
        assert grad.abs().sum() > 0
