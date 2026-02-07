"""Tests for the animation module (GIF export for training runs)."""

import json
import struct
import threading
from pathlib import Path

import pytest
import torch

from bezier_splatting.debug.animation import (
    AnimationConfig,
    FrameData,
    FrameRecorder,
    _compose_frame,
    _downsample_frames,
    _tensor_to_uint8,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def target_tensor():
    """Small solid-red target image (3, 16, 16)."""
    t = torch.zeros(3, 16, 16)
    t[0] = 1.0  # red channel
    return t


@pytest.fixture
def recorder(target_tensor):
    """FrameRecorder with 5 target frames for fast tests."""
    config = AnimationConfig(target_frames=5, fps=5)
    return FrameRecorder(config, target_tensor, H=16, W=16)


@pytest.fixture
def populated_recorder(recorder):
    """FrameRecorder with 5 captured frames."""
    for step in range(20):
        rendered = torch.rand(3, 16, 16)
        recorder.maybe_capture(
            step, 19, rendered,
            loss=1.0 / (step + 1), psnr=10.0 + step,
            n_open=3, n_closed=1,
        )
    return recorder


# ---------------------------------------------------------------------------
# AnimationConfig tests
# ---------------------------------------------------------------------------


class TestAnimationConfig:
    def test_defaults(self):
        cfg = AnimationConfig()
        assert cfg.layout == "standard"
        assert cfg.target_frames == 120
        assert cfg.fps == 10
        assert cfg.last_frame_hold == 3.0
        assert cfg.panel_size is None

    def test_custom_values(self):
        cfg = AnimationConfig(layout="full", target_frames=60, fps=20, panel_size=256)
        assert cfg.layout == "full"
        assert cfg.target_frames == 60
        assert cfg.fps == 20
        assert cfg.panel_size == 256


# ---------------------------------------------------------------------------
# Capture schedule tests
# ---------------------------------------------------------------------------


class TestDownsampleFrames:
    def _make_frames(self, n: int, event_at: set[int] | None = None) -> list[FrameData]:
        """Create n dummy FrameData objects."""
        event_at = event_at or set()
        return [
            FrameData(
                step=i * 10, rendered=torch.rand(3, 4, 4),
                loss=1.0 / (i + 1), psnr=10.0 + i,
                n_open=2, n_closed=1,
                event="prune" if i in event_at else None,
            )
            for i in range(n)
        ]

    def test_no_downsample_when_under_target(self):
        frames = self._make_frames(5)
        result = _downsample_frames(frames, target=10)
        assert len(result) == 5

    def test_downsamples_to_target(self):
        frames = self._make_frames(100)
        result = _downsample_frames(frames, target=20)
        assert len(result) <= 22  # target + first/last
        assert result[0].step == 0
        assert result[-1].step == frames[-1].step

    def test_preserves_topology_events(self):
        frames = self._make_frames(50, event_at={7, 23, 41})
        result = _downsample_frames(frames, target=10)
        event_steps = {f.step for f in result if f.event is not None}
        assert 70 in event_steps   # step = 7 * 10
        assert 230 in event_steps  # step = 23 * 10
        assert 410 in event_steps  # step = 41 * 10

    def test_keeps_first_and_last(self):
        frames = self._make_frames(50)
        result = _downsample_frames(frames, target=5)
        assert result[0].step == frames[0].step
        assert result[-1].step == frames[-1].step


# ---------------------------------------------------------------------------
# FrameRecorder capture tests
# ---------------------------------------------------------------------------


class TestFrameRecorderCapture:
    def test_captures_every_call(self, target_tensor):
        config = AnimationConfig(target_frames=5)
        rec = FrameRecorder(config, target_tensor, 16, 16)
        for step in range(20):
            rec.maybe_capture(
                step, 19, torch.rand(3, 16, 16),
                loss=0.5, psnr=20.0, n_open=2, n_closed=1,
            )
        # Captures every call; downsampling happens at export time
        assert rec.frame_count == 20

    def test_captures_sparse_calls(self, target_tensor):
        """Simulates the inspector calling every display_every steps."""
        config = AnimationConfig(target_frames=120)
        rec = FrameRecorder(config, target_tensor, 16, 16)
        total = 999
        display_every = max(1, total // 40)
        for step in range(total + 1):
            if step % display_every == 0 or step == total:
                rec.maybe_capture(
                    step, total, torch.rand(3, 16, 16),
                    loss=0.5, psnr=20.0, n_open=5, n_closed=2,
                )
        # Should have captured all ~41 callback calls, not just 2
        assert rec.frame_count >= 30

    def test_topology_event_recorded(self, recorder):
        recorder.record_topology_event(5, "prune")
        for step in range(20):
            recorder.maybe_capture(
                step, 19, torch.rand(3, 16, 16),
                loss=0.5, psnr=20.0, n_open=3, n_closed=1,
            )
        events = [f.event for f in recorder._frames if f.event is not None]
        assert "prune" in events

    def test_frame_data_fields(self, populated_recorder):
        frame = populated_recorder._frames[0]
        assert isinstance(frame, FrameData)
        assert isinstance(frame.step, int)
        assert isinstance(frame.loss, float)
        assert isinstance(frame.psnr, float)
        assert isinstance(frame.n_open, int)
        assert isinstance(frame.n_closed, int)
        assert frame.rendered.shape == (3, 16, 16)
        assert frame.rendered.device.type == "cpu"


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestFrameRecorderThreadSafety:
    def test_concurrent_capture(self, target_tensor):
        config = AnimationConfig(target_frames=200)
        rec = FrameRecorder(config, target_tensor, 16, 16)

        def worker(start, end):
            for s in range(start, end):
                rec.maybe_capture(
                    s, 999, torch.rand(3, 16, 16),
                    loss=0.1, psnr=25.0, n_open=5, n_closed=2,
                )

        threads = [
            threading.Thread(target=worker, args=(i * 250, (i + 1) * 250))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have captured frames without errors
        assert rec.frame_count > 0
        # No duplicate steps
        steps = [f.step for f in rec._frames]
        assert len(steps) == len(set(steps))


# ---------------------------------------------------------------------------
# Frame composition tests
# ---------------------------------------------------------------------------


class TestFrameComposition:
    @pytest.fixture
    def frame_data(self):
        return FrameData(
            step=50,
            rendered=torch.rand(3, 16, 16),
            loss=0.05,
            psnr=25.3,
            n_open=4,
            n_closed=2,
        )

    def test_minimal_layout(self, target_tensor, frame_data):
        target_np = _tensor_to_uint8(target_tensor)
        rendered_np = _tensor_to_uint8(frame_data.rendered)
        img = _compose_frame(
            "minimal", target_np, rendered_np, frame_data,
            total_steps=100, losses=[0.05], psnrs=[25.3], panel_size=None,
        )
        assert img.mode == "RGB"
        assert img.size[0] == 16  # width
        assert img.size[1] == 16  # height

    def test_standard_layout_width(self, target_tensor, frame_data):
        target_np = _tensor_to_uint8(target_tensor)
        rendered_np = _tensor_to_uint8(frame_data.rendered)
        img = _compose_frame(
            "standard", target_np, rendered_np, frame_data,
            total_steps=100, losses=[0.05], psnrs=[25.3], panel_size=None,
        )
        assert img.mode == "RGB"
        # 3 panels of 16px wide + 2 gaps of 2px = 52
        assert img.size[0] == 16 * 3 + 2 * 2

    def test_full_layout_has_chart(self, target_tensor, frame_data):
        target_np = _tensor_to_uint8(target_tensor)
        rendered_np = _tensor_to_uint8(frame_data.rendered)
        img_standard = _compose_frame(
            "standard", target_np, rendered_np, frame_data,
            total_steps=100, losses=[0.05, 0.04], psnrs=[25.3, 26.0],
            panel_size=None,
        )
        img_full = _compose_frame(
            "full", target_np, rendered_np, frame_data,
            total_steps=100, losses=[0.05, 0.04], psnrs=[25.3, 26.0],
            panel_size=None,
        )
        # Full layout should be taller than standard (has chart row)
        assert img_full.size[1] > img_standard.size[1]

    def test_panel_size_rescaling(self, target_tensor, frame_data):
        target_np = _tensor_to_uint8(target_tensor)
        rendered_np = _tensor_to_uint8(frame_data.rendered)
        img = _compose_frame(
            "standard", target_np, rendered_np, frame_data,
            total_steps=100, losses=[0.05], psnrs=[25.3], panel_size=128,
        )
        # 3 panels of 128px + 2 gaps of 2px = 388
        assert img.size[0] == 128 * 3 + 2 * 2

    def test_invalid_layout_raises(self, target_tensor, frame_data):
        target_np = _tensor_to_uint8(target_tensor)
        rendered_np = _tensor_to_uint8(frame_data.rendered)
        with pytest.raises(ValueError, match="Unknown layout"):
            _compose_frame(
                "bogus", target_np, rendered_np, frame_data,
                total_steps=100, losses=[], psnrs=[], panel_size=None,
            )


# ---------------------------------------------------------------------------
# GIF export tests
# ---------------------------------------------------------------------------


class TestGIFExport:
    def test_export_creates_gif(self, populated_recorder, tmp_path):
        gif_path = tmp_path / "test.gif"
        result = populated_recorder.export(gif_path)
        assert result == gif_path
        assert gif_path.exists()
        assert gif_path.stat().st_size > 0

    def test_gif_magic_bytes(self, populated_recorder, tmp_path):
        gif_path = tmp_path / "test.gif"
        populated_recorder.export(gif_path)
        with open(gif_path, "rb") as f:
            magic = f.read(6)
        # GIF magic: either GIF87a or GIF89a
        assert magic[:3] == b"GIF"
        assert magic[3:6] in (b"87a", b"89a")

    def test_sidecar_json(self, populated_recorder, tmp_path):
        gif_path = tmp_path / "test.gif"
        populated_recorder.export(gif_path)
        json_path = gif_path.with_suffix(".json")
        assert json_path.exists()

        meta = json.loads(json_path.read_text())
        assert meta["resolution"] == [16, 16]
        assert isinstance(meta["frames"], list)
        # Sidecar contains downsampled frames (target_frames=5), not raw count
        assert len(meta["frames"]) > 0
        assert len(meta["frames"]) <= populated_recorder.frame_count
        assert "step" in meta["frames"][0]
        assert "loss" in meta["frames"][0]
        assert "psnr" in meta["frames"][0]

    def test_empty_recorder_raises(self, recorder, tmp_path):
        gif_path = tmp_path / "empty.gif"
        with pytest.raises(ValueError, match="No frames captured"):
            recorder.export(gif_path)

    def test_export_creates_parent_dirs(self, populated_recorder, tmp_path):
        gif_path = tmp_path / "nested" / "dir" / "test.gif"
        result = populated_recorder.export(gif_path)
        assert result.exists()

    def test_all_layouts_export(self, target_tensor, tmp_path):
        for layout in ("minimal", "standard", "full"):
            config = AnimationConfig(layout=layout, target_frames=3, fps=5)
            rec = FrameRecorder(config, target_tensor, 16, 16)
            for step in range(10):
                rec.maybe_capture(
                    step, 9, torch.rand(3, 16, 16),
                    loss=0.5, psnr=20.0, n_open=2, n_closed=1,
                )
            gif_path = tmp_path / f"test_{layout}.gif"
            rec.export(gif_path)
            assert gif_path.exists()
            assert gif_path.stat().st_size > 0
