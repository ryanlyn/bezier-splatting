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
            closed_sampling_mode="official_cdf",
        )
        assert scene.closed_sampler.sampling_mode == "official_cdf"
