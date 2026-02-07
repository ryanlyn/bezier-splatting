"""Tests for fit_image loss presets and topology schedule modes."""

import math

import torch
import pytest

import bezier_splatting.optimization as opt_mod
from bezier_splatting.losses import LossConfig


def _patch_loss_to_capture(monkeypatch, bucket: dict):
    """Patch compute_loss to capture the effective LossConfig."""

    def _fake_compute_loss(rendered, target, scene, config, step, collect_loss_dict=False):
        bucket["config"] = config
        zero_loss = rendered.mean() * 0.0
        return zero_loss, {"reconstruction": 0.0, "total": 0.0}

    monkeypatch.setattr(opt_mod, "compute_loss", _fake_compute_loss)


class TestLossPresetResolution:
    def test_closed_preset_is_default(self, monkeypatch):
        captured: dict = {}
        _patch_loss_to_capture(monkeypatch, captured)

        target = torch.rand(3, 16, 16)
        _ = opt_mod.fit_image(target, n_open=1, n_closed=1, steps=1, log_every=1)

        cfg = captured["config"]
        assert isinstance(cfg, LossConfig)
        assert cfg.apply_shape_reg is True
        assert cfg.apply_opacity_prior is True
        assert cfg.apply_curvature is True
        assert cfg.apply_boundary is True

    def test_minimal_preset_disables_regularizers(self, monkeypatch):
        captured: dict = {}
        _patch_loss_to_capture(monkeypatch, captured)

        target = torch.rand(3, 16, 16)
        _ = opt_mod.fit_image(
            target,
            n_open=1,
            n_closed=1,
            steps=1,
            log_every=1,
            loss_preset="minimal",
            lambda_xing=0.0,
        )

        cfg = captured["config"]
        assert cfg.apply_shape_reg is False
        assert cfg.apply_opacity_prior is False
        assert cfg.apply_curvature is False
        assert cfg.apply_boundary is False

    def test_custom_preset_requires_explicit_config(self):
        target = torch.rand(3, 16, 16)
        with pytest.raises(ValueError, match="requires an explicit loss_config"):
            _ = opt_mod.fit_image(
                target,
                n_open=1,
                n_closed=1,
                steps=1,
                loss_preset="custom",
            )

    def test_explicit_loss_config_overrides_preset(self, monkeypatch):
        captured: dict = {}
        _patch_loss_to_capture(monkeypatch, captured)

        custom = LossConfig(
            apply_shape_reg=False,
            apply_opacity_prior=False,
            apply_curvature=False,
            apply_boundary=False,
            lambda_xing=0.2,
        )
        target = torch.rand(3, 16, 16)
        _ = opt_mod.fit_image(
            target,
            n_open=1,
            n_closed=1,
            steps=1,
            log_every=1,
            loss_preset="closed",
            loss_config=custom,
        )
        assert captured["config"] is custom


class TestTopologyScheduleModes:
    def test_alternating_runs_prune_and_densify_in_separate_phases(self, monkeypatch):
        calls: list[tuple[bool, bool, int | None]] = []

        _patch_loss_to_capture(monkeypatch, {})

        def _fake_topology(
            scene,
            target,
            rendered,
            step,
            total_steps,
            H,
            W,
            config,
            optimizer,
            do_prune=True,
            do_densify=True,
            densify_n_override=None,
        ):
            calls.append((do_prune, do_densify, densify_n_override))
            # Simulate pruning 3 curves on prune-only phases.
            if do_prune and not do_densify:
                return False, 3
            return False, 0

        monkeypatch.setattr(opt_mod, "_prune_and_densify", _fake_topology)

        target = torch.rand(3, 16, 16)
        _ = opt_mod.fit_image(
            target,
            n_open=1,
            n_closed=0,
            steps=7,
            prune_every=2,
            prune_stop_before_end=0,
            topology_schedule="alternating",
            topology_start_step=0,
            topology_max_step_open=100,
            log_every=1000,
        )

        assert calls[0] == (True, False, None)   # prune phase
        assert calls[1] == (False, True, 3)      # densify phase with pending count
        assert calls[2] == (True, False, None)   # next prune phase

    def test_unified_schedule_runs_combined_phase(self, monkeypatch):
        calls: list[tuple[bool, bool]] = []
        _patch_loss_to_capture(monkeypatch, {})

        def _fake_topology(
            scene,
            target,
            rendered,
            step,
            total_steps,
            H,
            W,
            config,
            optimizer,
            do_prune=True,
            do_densify=True,
            densify_n_override=None,
        ):
            calls.append((do_prune, do_densify))
            return False, 0

        monkeypatch.setattr(opt_mod, "_prune_and_densify", _fake_topology)

        target = torch.rand(3, 16, 16)
        _ = opt_mod.fit_image(
            target,
            n_open=1,
            n_closed=0,
            steps=7,
            prune_every=2,
            prune_stop_before_end=0,
            topology_schedule="unified",
            log_every=1000,
        )

        assert all(p == (True, True) for p in calls)
        assert len(calls) == 3  # steps 2, 4, 6


class TestCallbackLossSyncControl:
    def test_callback_can_skip_per_step_loss_values(self):
        observed: list[tuple[int, float]] = []

        def _callback(step: int, loss: float, _scene):
            observed.append((step, loss))
            return None

        target = torch.rand(3, 16, 16)
        _ = opt_mod.fit_image(
            target,
            n_open=1,
            n_closed=0,
            steps=5,
            log_every=3,
            callback=_callback,
            callback_requires_loss=False,
        )

        finite_steps = [step for step, loss in observed if math.isfinite(loss)]
        assert finite_steps == [0, 3, 4]

    def test_callback_defaults_to_per_step_loss_values(self):
        observed: list[tuple[int, float]] = []

        def _callback(step: int, loss: float, _scene):
            observed.append((step, loss))
            return None

        target = torch.rand(3, 16, 16)
        _ = opt_mod.fit_image(
            target,
            n_open=1,
            n_closed=0,
            steps=5,
            log_every=3,
            callback=_callback,
        )

        finite_steps = [step for step, loss in observed if math.isfinite(loss)]
        assert finite_steps == [0, 1, 2, 3, 4]
