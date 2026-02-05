# CLAUDE.md — Bézier Splatting

Pure PyTorch reimplementation of "Bézier Splatting for Fast and Differentiable Vector Graphics Rendering" (NeurIPS 2025, arxiv 2503.16424). Educational clarity over raw speed. No CUDA kernels.

## Quick Reference

```bash
uv run pytest tests/ -v                              # full suite (includes slow reconstruction)
uv run pytest tests/ -v --ignore=tests/test_reconstruction.py  # fast unit tests only (~7s)
uv run pytest tests/test_reconstruction.py -k circle  # single reconstruction target
uv run pytest tests/ --save-outputs                   # save diagnostic images to tests/outputs/
uv run marimo edit notebooks/vectorize.py             # interactive notebook
```

## Architecture

```
src/bezier_splatting/
├── bezier.py        # Pure Bézier math (evaluate, tangent, Bernstein basis)
├── sampling.py      # Curve → GaussianParams conversion (open + closed samplers)
├── rasterizer.py    # Tile-based 2D Gaussian splatting (front-to-back alpha compositing)
├── model.py         # VectorGraphicsScene (nn.Module combining all primitives)
├── optimization.py  # Training loop + LIVE Xing loss + pruning/densification
├── metrics.py       # MSE, PSNR, SSIM, edge MSE
└── svg.py           # SVG export
```

**Data flow:** `model.forward()` → `sampling` (curves → Gaussians) → `rasterizer` (Gaussians → image)

**Dependency direction:** `bezier ← sampling ← model → rasterizer`. `optimization` orchestrates `model`. `svg` reads from `model`. No circular deps.

## Critical Design Decisions

### Normalized [0, 1] coordinates
All control points are stored in [0, 1] and scaled to pixel space at render time via `(H, W)` args. This makes the scene resolution-independent. Every function that touches CPs in pixel space receives explicit `H, W` parameters — grep for `* scale` or `* torch.tensor([W, H]` to find scaling sites.

### Open curves = 3 connected cubics
Each open curve has 10 control points. Split into 3 cubic segments sharing endpoints at indices [3] and [6]: segment 0 = CPs[0:4], segment 1 = CPs[3:7], segment 2 = CPs[6:10].

### Per-segment opacity (open curves)
Open curves have shape `(N, 3)` opacities — one per cubic segment (paper Appendix D). The `composite_segment_sizes()` function in `bezier.py` is the **single source of truth** for how K samples are distributed across 3 segments. Both `evaluate_composite_bezier` and `OpenCurveSampler` use it. Extras go to earlier segments: K=20 → [7, 7, 6].

### Closed curves = paired boundaries
Each closed curve has 2 boundary curves `(N, 2, num_cp, 2)` with shared first/last CPs. `_enforce_shared_endpoints()` averages them each forward pass. R+2 intermediate curves are interpolated between boundaries (R intermediate + 2 boundaries, paper Eq. 6).

### Per-curve depth sorting
Depth is per-curve, not per-Gaussian. All Gaussians from the same curve share a `curve_id`. Smallest-area curves are frontmost (index 0 in front-to-back compositing). Area proxy: polyline length × stroke width (open), bounding box area (closed).

### Scale clamping
Both samplers clamp σ_x and σ_y to `min=0.1` pixels. Without this, shared endpoints where boundaries pinch together produce near-zero scales → degenerate covariance matrices → NaN gradients during backward pass.

### Safe norms
All norm computations use `torch.sqrt(x**2 + 1e-12)` instead of `torch.norm()` to prevent NaN gradients at zero vectors.

## Parameter Shapes & Ranges

| Parameter | Shape | Space | Notes |
|-----------|-------|-------|-------|
| `open_control_points` | `(N, 10, 2)` | [0, 1] | Normalized coords |
| `open_colors` | `(N, 3)` | unconstrained | `sigmoid()` applied at render time |
| `open_opacities` | `(N, 3)` | pre-sigmoid | One per cubic segment |
| `open_stroke_widths` | `(N,)` | pre-sigmoid | Maps to [0.5, 5] px via `0.5 + sigmoid(w) * 4.5` |
| `closed_boundary_cp` | `(N, 2, num_cp, 2)` | [0, 1] | 2 boundaries, shared endpoints |
| `closed_colors` | `(N, 3)` | unconstrained | `sigmoid()` applied at render time |
| `closed_opacities` | `(N,)` | pre-sigmoid | Single opacity per closed curve |

## Optimization Details

- **Optimizer:** Adam with per-parameter-group learning rates
  - Control points: 1e-3 (normalized coords, so ~0.25 px/step at 256×256)
  - Colors: 0.01
  - Opacities: 0.1
  - Stroke widths: 0.05
- **LR decay:** StepLR with `lr_decay` accumulator that survives optimizer rebuilds after pruning
- **Loss:** `MSE(rendered, target) + λ_xing * L_xing`
- **Xing loss:** LIVE direction-gated sine penalty on cubic control polygons — prevents self-intersection
- **Pruning:** every 400 steps, stops 1000 steps before end. Removes low-opacity + small-area curves
- **Densification:** inserts new curves at high-error regions. Open vs closed chosen by error region aspect ratio. Color initialized from target in logit space (inverse sigmoid)

## Test Structure

- `test_bezier.py` — Bernstein basis properties, endpoint interpolation, tangent direction, composite continuity
- `test_sampling.py` — Output shapes/counts, positive scales, curve IDs, segment opacity alignment (parametrized over K), pixel-space means
- `test_rasterizer.py` — Covariance math, 2x2 inverse, empty/single/multi Gaussian rendering, tile boundary continuity, compositing order
- `test_gradients.py` — Gradient flow to all parameter types, finite-difference vs autograd comparison
- `test_reconstruction.py` — Full optimization on programmatic targets (circle, overlap, strokes, gradient, composition). Tier 1 (must pass) and Tier 2 (quality gate) thresholds. **Slow** — each target runs a full training loop.
- `conftest.py` — `--save-outputs` flag for diagnostic images

## Common Gotchas

1. **Colors are pre-sigmoid.** `model.forward()` applies `torch.sigmoid(self.open_colors)` before passing to samplers. Don't double-sigmoid.
2. **Opacities are pre-sigmoid too.** The rasterizer applies sigmoid internally.
3. **Open opacity shape is (N, 3), not (N,).** One per cubic segment. Closed opacity is (N,).
4. **Empty curves.** When n_open=0 or n_closed=0, parameters are registered as buffers (not Parameters). Check `scene.n_open > 0` before accessing `.grad`.
5. **After pruning, optimizer must be rebuilt** because parameters are replaced with new tensors. `fit_image` handles this with `_build_param_groups()`.
6. **Reconstruction tests are slow.** Use `--ignore=tests/test_reconstruction.py` for fast iteration.

## Changelog

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
