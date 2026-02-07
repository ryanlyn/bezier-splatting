"""Debug utilities for Bezier Splatting optimization."""

from .tracker import DebugTracker
from .collectors import collect_gradient_stats, collect_curve_stats, snapshot_scene
from .assertions import check_health
from .checkpoints import save_checkpoint, load_checkpoint, list_checkpoints
from .animation import AnimationConfig, FrameRecorder
from .viz import compute_error_map, make_loss_chart
