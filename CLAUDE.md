# CLAUDE.md — Bézier Splatting

Pure PyTorch reimplementation of "Bézier Splatting for Fast and Differentiable Vector Graphics Rendering" (NeurIPS 2025, arxiv 2503.16424). Educational clarity over raw speed. No CUDA kernels.

## Quick Reference

```bash
uv run pytest tests/ -v --ignore=tests/test_reconstruction.py  # unit tests only (~7s)
uv run pytest tests/ -v --typecheck --ignore=tests/test_reconstruction.py  # with shape checking
uv run pytest tests/test_reconstruction.py --fast -v  # fast reconstruction (~6 min)
uv run pytest tests/ -v                              # full suite (~25 min)
uv run pytest tests/ --save-outputs                   # save diagnostic images to tests/outputs/
uv run pytest tests/test_reconstruction.py -k circle  # single reconstruction target
uv run marimo edit notebooks/vectorize.py             # interactive notebook
```

## Architecture

```
src/bezier_splatting/
├── bezier.py        # Pure Bézier math (evaluate, tangent, Bernstein basis)
├── coords.py        # Coordinate space conversion: [-1,1] model ↔ pixel ↔ legacy [0,1]
├── sampling.py      # Curve → GaussianParams conversion (open + closed samplers)
├── rasterizer.py    # Chunked batched tile rendering for 2D Gaussian splatting (front-to-back alpha compositing)
├── model.py         # VectorGraphicsScene (nn.Module combining all primitives, learned _depth)
├── optimization.py  # Training loop + optimizer state surgery + pruning/densification
├── losses.py        # Configurable composite loss system (LossConfig + 6 loss terms)
├── topology.py      # Pruning heuristics (PruneConfig) + densification center computation
├── adan.py          # Adan optimizer (Adaptive Nesterov Momentum, pluggable via optimizer_type="adan")
├── area.py          # True enclosed area for closed curves (depth sorting)
├── metrics.py       # MSE, PSNR, SSIM, edge MSE
└── svg.py           # SVG export
```

**Data flow:** `model.forward()` → `sampling` (curves → Gaussians) → `rasterizer` (Gaussians → image)

**Dependency direction:** `bezier ← sampling ← model → rasterizer`. `optimization` orchestrates `model`, `losses`, `topology`, `adan`. `coords` is a leaf dependency used by `sampling`, `model`, `optimization`, `topology`, `svg`. No circular deps.

## Critical Design Decisions

### Normalized [-1, 1] coordinates
All control points are stored in [-1, 1] and scaled to pixel space at render time via `(H, W)` args. This makes the scene resolution-independent. The `coords.py` module provides `model_to_pixel()` and `pixel_to_model()` — the single source of truth for coordinate conversion. A `legacy_to_model()` helper exists for converting old [0, 1] checkpoints. Every function that touches CPs in pixel space receives explicit `H, W` parameters.

### Open curves = 3 connected cubics
Each open curve has 10 control points. Split into 3 cubic segments sharing endpoints at indices [3] and [6]: segment 0 = CPs[0:4], segment 1 = CPs[3:7], segment 2 = CPs[6:10].

### Per-segment opacity (open curves)
Open curves have shape `(N, 3)` opacities — one per cubic segment (paper Appendix D). The `composite_segment_sizes()` function in `bezier.py` is the **single source of truth** for how K samples are distributed across 3 segments. Both `evaluate_composite_bezier` and `OpenCurveSampler` use it. Extras go to earlier segments: K=20 → [7, 7, 6].

### Closed curves = paired boundaries
Each closed curve has 2 boundary curves with shared first/last CPs. Endpoints are stored once in `closed_shared_pts` (N, 2, 2) and interior CPs in `closed_interior_cp` (N, 2, num_cp-2, 2). `_assemble_boundary_cp()` reconstructs the full (N, 2, num_cp, 2) tensor by broadcasting shared endpoints to both boundaries — structurally impossible for them to disagree. R+2 intermediate curves are interpolated between boundaries (R intermediate + 2 boundaries, paper Eq. 6).

### Learned depth with heuristic overwrite
Depth is a learned `nn.Parameter` (`_depth`, shape `(N_total, 1)`) storing pre-sigmoid values. Layout is `[open_curves | closed_curves]`. Heuristic overwrites keep depth aligned with geometric area:
- **Closed curves:** AABB area overwritten every forward pass (always stays geometric).
- **Open curves:** polyline length x stroke width overwritten every 20 steps for the first 10k iterations, then frozen to let the optimizer fine-tune.
- The `get_depth` property applies sigmoid, returning values in [0, 1]. Smaller depth = frontmost (index 0 in compositing after argsort).

### Mode-specific scale clamping
Scale clamping is split by role to prevent NaN gradients from degenerate covariance matrices:
- **Open curves:** σ_x gets spacing/rho + 0.5 offset (always positive). σ_y is stroke width (sigmoid to [0.5, 5] px).
- **Closed curve boundaries** (indices 0, 1): σ_x clamped to min=0.3, σ_y clamped to [0.75, 1.0]. Boundary taper attenuates σ_x at first/last 3 samples (factors [0.4, 0.9, 1.0]) to soften endpoints.
- **Closed curve interiors** (indices 2+): Ratio mutual clamp — when σ_y < 0.1, both scales clamped to 3:1 max ratio. Safety floor of min=0.1 for both.
- Without clamping, shared endpoints where boundaries pinch together produce near-zero scales → degenerate covariance → NaN backward.

### Chunked tile rendering
The rasterizer processes tiles in chunks rather than individually. Tiles are sorted by Gaussian count (ascending) and processed in groups of `chunk_size` (default 16). Within each chunk, tiles are padded to the chunk-local max Gaussian count, enabling batched einsum/cumprod/compositing. Padding entries have zeroed opacity so they don't contribute. Sensitivity analysis shows a U-shaped curve: chunk_size=1 has too much Python loop overhead, chunk_size=256 has too much memory pressure from intermediate tensors. Sweet spot is 8-16 tiles per chunk.

### Safe norms
All norm computations use `torch.sqrt(x**2 + 1e-12)` instead of `torch.norm()` to prevent NaN gradients at zero vectors.

## Parameter Shapes & Ranges

| Parameter | Shape | Space | Notes |
|-----------|-------|-------|-------|
| `open_control_points` | `(N, 10, 2)` | [-1, 1] | Normalized coords |
| `open_colors` | `(N, 3)` | unconstrained | `sigmoid()` applied at render time |
| `open_opacities` | `(N, 3)` | pre-sigmoid | One per cubic segment |
| `open_stroke_widths` | `(N,)` | pre-sigmoid | Maps to [0.5, 5] px via `0.5 + sigmoid(w) * 4.5` |
| `closed_shared_pts` | `(N, 2, 2)` | [-1, 1] | Shared [start, end] × [x, y] for both boundaries |
| `closed_interior_cp` | `(N, 2, num_cp-2, 2)` | [-1, 1] | Interior CPs per boundary (excludes shared endpoints) |
| `closed_colors` | `(N, 3)` | unconstrained | `sigmoid()` applied at render time |
| `closed_opacities` | `(N,)` | pre-sigmoid | Single opacity per closed curve |
| `_depth` | `(N_total, 1)` | pre-sigmoid | Layout: [open \| closed]. `get_depth` applies sigmoid → [0, 1] |

## Optimization Details

- **Optimizer:** Adam (default) with per-parameter-group learning rates. Adan available via `optimizer_type="adan"` (betas=(0.98, 0.92, 0.99), 4 state tensors).
  - Control points: `0.25 / max(H, W)` (resolution-scaled so ~0.25 px displacement/step)
  - Colors: 0.01
  - Opacities: 0.1
  - Stroke widths: 0.05
  - Depth: 0.0 (heuristically overwritten, not optimized)
- **LR decay:** StepLR with `lr_decay` accumulator that survives optimizer state surgery after pruning
- **Loss:** Configurable via `LossConfig`. Default: `MSE(rendered, target) + λ_xing * L_xing`. Available terms: L2/L1/Fusion1 reconstruction, Xing loss, shape regularizer, opacity prior, curvature loss, boundary joint loss. When `loss_config=None`, backward-compatible defaults disable all regularizers.
- **Xing loss:** LIVE direction-gated sine penalty on cubic control polygons — prevents self-intersection. Moved to `losses.py`.
- **Pruning:** Configurable via `PruneConfig` in `topology.py`. Heuristics: outside-image ratio, overlap+color suppression, tiny curve removal, staged opacity thresholds, staged area thresholds. Every 400 steps, stops 1000 steps before end.
- **Densification:** inserts new curves at high-error regions. Open vs closed chosen by error region aspect ratio. Color initialized from target in logit space (inverse sigmoid).
- **Optimizer state surgery:** On prune/densify, optimizer momentum tensors are sliced/extended in-place via `_prune_optimizer_state()`, `_extend_optimizer_state()`, `_splice_optimizer_state()`. Full optimizer rebuild only when a 0→n curve type transition introduces new param groups.

## Test Structure

- `test_bezier.py` — Bernstein basis properties, endpoint interpolation, tangent direction, composite continuity
- `test_sampling.py` — Output shapes/counts, positive scales, curve IDs, segment opacity alignment (parametrized over K), pixel-space means
- `test_rasterizer.py` — Covariance math, 2x2 inverse, empty/single/multi Gaussian rendering, tile boundary continuity, compositing order
- `test_gradients.py` — Gradient flow to all parameter types, finite-difference vs autograd comparison
- `test_losses.py` — Reconstruction losses (L2/L1/Fusion1), shape regularizer, opacity prior, curvature loss, boundary joint loss, Xing loss, composite `compute_loss`, LossConfig defaults
- `test_topology.py` — AABB computation, outside ratio, pairwise IoU, tiny curve mask, overlap suppression, full prune masks (open + closed), densify centers, color distance
- `test_adan.py` — Convergence, state tensors, weight decay, state dict roundtrip, per-group LR, invalid hyperparameter validation
- `test_optimizer_surgery.py` — Prune/extend/splice state surgery for both Adan and Adam, value preservation, shape correctness, continued training after surgery
- `test_svg.py` — RGB string formatting, open/closed curve path generation, coordinate round-trips, scene-level SVG export, depth ordering, opacity handling, dimension defaults
- `test_reconstruction.py` — Full optimization on programmatic targets (circle, overlap, strokes, gradient, composition). Tier 1 (must pass) and Tier 2 (quality gate) thresholds. **Slow** — each target runs a full training loop.
- `conftest.py` — `--save-outputs` flag for diagnostic images, `--fast` flag for reduced reconstruction suite

## Common Gotchas

1. **Colors are pre-sigmoid.** `model.forward()` applies `torch.sigmoid(self.open_colors)` before passing to samplers. Don't double-sigmoid.
2. **Opacities are pre-sigmoid too.** The rasterizer applies sigmoid internally.
3. **Open opacity shape is (N, 3), not (N,).** One per cubic segment. Closed opacity is (N,).
4. **Empty curves.** When n_open=0 or n_closed=0, parameters are registered as buffers (not Parameters). Check `scene.n_open > 0` before accessing `.grad`.
5. **`closed_boundary_cp` is a read-only property**, not a parameter. It assembles from `closed_shared_pts` + `closed_interior_cp`. For optimizer param groups and pruning/densification, use the two underlying tensors directly.
6. **After pruning, optimizer state is surgically updated** (momentum tensors sliced/extended) rather than rebuilt from scratch. Full rebuild only happens on 0→n curve type transitions. The surgery functions live in `optimization.py`.
7. **Reconstruction tests are slow.** Use `--ignore=tests/test_reconstruction.py` for fast iteration.
8. **Never add `from __future__ import annotations`.** It stringifies annotations at parse time, breaking jaxtyping's runtime shape introspection. All source files use Python 3.11+ native syntax (`X | Y`, `list[...]`) instead.
9. **Chunk size sensitivity.** The rasterizer's `chunk_size` default (16) was chosen via sensitivity analysis. Too small = Python loop overhead; too large = memory pressure from padded intermediate tensors. The optimal value shifts slightly lower for denser scenes.
10. **`_depth` layout is `[open | closed]`.** When pruning or densifying, the depth parameter must be spliced to match the new curve layout. The `_splice_optimizer_state()` function handles inserting new open curve entries between existing open and closed entries.
11. **`LossConfig` defaults disable regularizers.** When `loss_config=None` is passed to `fit_image`, a backward-compatible default is created with only MSE + Xing enabled. Enable regularizers explicitly via `LossConfig(apply_shape_reg=True, ...)`.
12. **Coordinate space is [-1, 1], not [0, 1].** Use `coords.model_to_pixel()` and `coords.pixel_to_model()` for all conversions. Clamping bounds for control points are -1/+1, not 0/1.

## Runtime Shape Checking (jaxtyping + beartype)

Type annotations use jaxtyping shape specs (`Float[Tensor, "N 10 2"]`) checked at runtime via beartype. Activated only during unit testing via pytest import hook — zero overhead in normal usage and reconstruction tests.

### Usage
- `--typecheck` flag activates the import hook for `bezier_splatting.*`
- Only use with unit tests, NOT reconstruction tests
- Annotations live in source files as standard Python type hints — useful documentation regardless of checking

### Dimension naming convention
| Symbol | Meaning | Literal? |
|--------|---------|----------|
| `N` | curves | named |
| `K` | samples/curve | named |
| `G` | total Gaussians | named |
| `H`, `W` | image dims | named |
| `CP` | control points | named |
| `C` | color channels | named |
| `M1` | degree + 1 (Bernstein) | named |
| `3`, `10`, `2` | fixed sizes | literal (enforced) |

## Changelog

### 2026-02-06 (v2)
- **Coordinate space changed from [0, 1] to [-1, 1].** New `coords.py` module provides `model_to_pixel()`, `pixel_to_model()`, and `legacy_to_model()` as single source of truth. All CP initialization, clamping, and conversion sites updated. CP learning rate is now `0.25 / max(H, W)` (resolution-scaled for the wider range).
- **Configurable composite loss system.** New `losses.py` with `LossConfig` dataclass + 6 loss terms: L2/L1/Fusion1 reconstruction, Xing loss (moved from optimization.py), shape regularizer, opacity prior, curvature loss, boundary joint loss. All individually weighted and toggleable. Default config preserves backward compatibility (only MSE + Xing).
- **Richer pruning heuristics.** New `topology.py` with `PruneConfig` dataclass. Heuristics: outside-image ratio, overlap + color similarity suppression, tiny curve removal, staged opacity thresholds (early vs late training), staged area thresholds (3 phases). All pure computation — no in-place scene mutation.
- **Optimizer state surgery.** Prune/extend/splice operations on optimizer momentum tensors (`_prune_optimizer_state`, `_extend_optimizer_state`, `_splice_optimizer_state`) preserve accumulated momentum for surviving curves instead of resetting to zero. Full optimizer rebuild only on 0→n curve type transitions.
- **Adan optimizer.** New `adan.py` implements Adaptive Nesterov Momentum (Xie et al., 2023) with three momentum terms. Pluggable via `optimizer_type="adan"` (Adam is the default).
- **Learned depth parameter.** `_depth` is now an `nn.Parameter` (shape `(N_total, 1)`, layout `[open | closed]`) with heuristic overwrite schedule: closed curve AABB area overwritten every forward pass; open curve polyline area overwritten every 20 steps for first 10k iterations then frozen. `get_depth` property applies sigmoid → [0, 1].
- **Mode-specific scale clamping.** Closed curve boundary scales get tighter clamps (σ_x min=0.3, σ_y in [0.75, 1.0]) with endpoint taper. Interior scales get ratio mutual clamp (3:1 max when σ_y < 0.1) plus min=0.1 safety floor.
- **26 new SVG export tests** in `test_svg.py`. Covers RGB formatting, coordinate round-trips, segment structure, opacity handling, depth ordering, dimension defaults.
- **New test files:** `test_losses.py` (28 tests), `test_topology.py` (22 tests), `test_adan.py` (13 tests), `test_optimizer_surgery.py` (17 tests), `test_svg.py` (26 tests). Total unit tests: 193.

### 2026-02-06 (v1)
- **Vectorized rasterizer tile loop.** Replaced Python triple-loop tile assignment (4× CPU-GPU syncs) with GPU-side boolean overlap matrix. Replaced per-tile rendering loop (256 iterations) with chunked batched rendering — tiles sorted by Gaussian count, processed in chunks of 16 with padded gather + batched einsum/cumprod. Pre-built pixel grid eliminates per-tile allocation. ~1.5× speedup on forward+backward (308→205 ms at 256×256 with 20 open + 5 closed curves).
- **Detached depth sorting from autograd.** Wrapped area computation + argsort in `torch.no_grad()` since areas only feed discrete sort indices. Gaussian reordering stays outside the block to preserve gradient flow.
- **Sampling quick wins.** Added `compute_tangents=False` to `evaluate_composite_bezier` (skips 3× unused `bezier_tangent` calls). Cached Bernstein binomial coefficients for degrees 1-3 in `_BINOM_COEFFS`. Deferred `loss.item()` to logging/callback steps only.

### 2025-02-06
- **Added jaxtyping + beartype runtime shape checking.** All core source files annotated with `Float[Tensor, "N K 2"]`-style shape specs. Activated via `--typecheck` pytest flag using jaxtyping's import hook — zero overhead in normal usage. Removed `from __future__ import annotations` from all files (incompatible with runtime introspection).
- **Added `--fast` flag for reconstruction tests.** Runs only `circle` + `strokes` targets at tier-1 with halved steps (~2.5 min vs ~25 min full suite). Covers both open and closed curve samplers.
- **Added optimization cache** to `test_reconstruction.py`. Tier-1 and tier-2 now share the same `fit_image()` run per target via `_optimization_cache` dict, halving full-suite time.
- **Hard shared-endpoint constraint** for closed curves. Replaced single `closed_boundary_cp` parameter with `closed_shared_pts` (N, 2, 2) + `closed_interior_cp` (N, 2, num_cp-2, 2). Both boundaries now read endpoints from the same tensor — structurally impossible for them to disagree. Deleted `_enforce_shared_endpoints()` averaging. Added `closed_boundary_cp` read-only property for backward compatibility.
- **Seeded reconstruction tests** for reproducibility. `target_config` fixture now calls `torch.manual_seed(hash(name) % 2**32)` before each target, eliminating flaky results from random initialization. Reverted circle steps back to 1000 (was temporarily bumped to 1500).

### 2025-02-05
- **Reduced reconstruction test resolution** to 64×64 (was 256×256) for faster CI (~50 min vs ~5 hours). Adjusted curve counts and steps proportionally. Added `TEST_RESOLUTION` constant.
- **Switched rotation to central differences** (paper Eq. 8). Replaced analytic `bezier_tangent` → `atan2` with `_central_diff_angles()` using `(X[k+1]-X[k-1])` for interior, forward/backward diff at boundaries. Representation-agnostic — will work with future non-Bézier curve types and aligns with CUDA kernel design. Extracted as shared helper used by both samplers.
- **Fixed per-segment opacity mismatch** (sampling.py). Old code used `[K//3, K//3, K-2*(K//3)]` bins which didn't match `evaluate_composite_bezier`'s remainder distribution. Extracted `composite_segment_sizes()` into `bezier.py` as single source of truth.
- **Added `test_segment_opacity_alignment`** — parametrized test over K=9,10,11,19,20,21 verifying opacity bins match curve sampling.

### 2025-02-04
- **Rewrote optimization.py**: LIVE Xing loss, StepLR decay with accumulator, normalized-coord lr (1e-3), 3-opacity support, both open+closed densification, color init from target in logit space, all-pruned edge case handling.
- **Rewrote svg.py**: [0,1] → pixel coord scaling, per-segment opacity → mean for SVG export.
- **Fixed NaN gradients in closed curves**: Added `.clamp(min=0.1)` to σ_x and σ_y in both samplers. Shared endpoints where boundaries pinch together produced near-zero scales → degenerate covariance → NaN backward.
- **Safe norms**: Replaced `torch.norm(diffs, dim=-1)` with `torch.sqrt((diffs**2).sum(-1) + 1e-12)` in sampling.py.
- **Updated all test files** for new API: curve_ids in GaussianParams, (N,3) opacities, H/W args, R+2 closed curve counts.
- Pipeline convergence verified: PSNR 26.64, SSIM 0.899 on red circle (64×64, 1000 steps).

### 2025-02-03
- **Rewrote sampling.py**: Normalized [0,1] CPs with H/W scaling, per-segment opacity (N,3), boundary-biased spacing for closed curves, composite Bézier evaluation.
- **Rewrote model.py**: [0,1] coords, shared endpoint enforcement, per-curve depth sorting by area, curve_ids for depth grouping.
- Initial implementation of all modules: bezier.py, rasterizer.py, metrics.py, notebooks/vectorize.py.
