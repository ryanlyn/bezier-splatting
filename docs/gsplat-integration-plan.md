# Plan: Integrate `gsplat` (or another high-performance CUDA Gaussian splatting backend)

## 1) Goals and constraints

- Keep the existing API and training flow intact:
  - `VectorGraphicsScene.forward(...)` currently routes rendering through `rasterize(...)` with backend/tile/chunk overrides.
  - `fit_image(...)` exposes raster backend config and stores it in the scene.
- Preserve differentiability and depth-ordered alpha compositing semantics used by the current pure-PyTorch renderer.
- Add CUDA acceleration as an optional backend, not a hard dependency, so CPU/MPS workflows continue to work.

## 2) Candidate backend strategy

### Preferred path: `gsplat` backend

- Add a new backend key (e.g. `"cuda_gsplat"`) to backend selection.
- Map our `GaussianParams` to `gsplat` inputs in a dedicated adapter layer:
  - means (2D) -> backend positions
  - scales + rotations -> covariance/conic representation expected by kernel
  - opacities -> alpha logits/probabilities
  - colors -> per-Gaussian RGB
  - depth ordering -> explicit sort order before kernel invocation (or pass depth if kernel supports z-order compositing).

### Fallback path: custom CUDA extension

- If `gsplat` API/feature mismatch appears (e.g. strict 3D assumptions), implement a lightweight custom extension with:
  - fused tile binning + raster + compositing
  - forward + backward kernels
  - PyTorch C++/CUDA bindings
- Keep the same Python backend contract so swapping implementations remains transparent.

## 3) Codebase integration points

1. **Backend enum/validation** in `src/bezier_splatting/rasterizer.py`
   - Extend `_resolve_backend(...)` and public `rasterize(...)` dispatch to include CUDA backend(s).
2. **Scene/model plumbing** in `src/bezier_splatting/model.py`
   - Keep `VectorGraphicsScene.forward(...)` call signature stable.
   - Optionally add backend-specific kwargs (safe defaults) without breaking existing calls.
3. **Optimization API propagation** in `src/bezier_splatting/optimization.py`
   - Ensure `fit_image(...)` can select CUDA backend and pass backend options to `scene(...)` calls.
4. **Packaging/dependency gates** in `pyproject.toml`
   - Add optional dependency group, e.g. `[project.optional-dependencies].cuda = ["gsplat>=..."]`.
   - Use lazy import + clear error messaging when backend requested but dependency missing.

## 4) Implementation phases

### Phase A — Interface scaffolding (no functional CUDA yet)

- Add backend identifier(s): `"cuda_gsplat"` (and optionally `"cuda_custom"`).
- Create adapter function skeletons and runtime capability checks:
  - device must be CUDA
  - package import available
  - dtype support validated
- Add user-facing error messages with actionable remediation.

### Phase B — `gsplat` adapter implementation

- Implement Gaussian parameter conversion utilities in a focused module (e.g. `src/bezier_splatting/cuda_backends/gsplat_adapter.py`).
- Implement raster call path and return tensor shape parity `(3, H, W)`.
- Validate numerical parity vs `reference` backend on representative scenes:
  - empty scene
  - open-only
  - closed-only
  - mixed-depth overlap cases
  - high-opacity stress case.

### Phase C — performance tuning

- Benchmark against current `reference` and `mps` paths (`benchmarks/benchmark_rasterizer.py`):
  - wall-clock forward
  - forward+backward
  - peak memory
  - scaling with Gaussian count and resolution.
- Tune batching/sort/precision decisions:
  - mixed precision (fp16/bf16) where safe
  - persistent buffers/caching
  - tile sizing parameters if exposed.

### Phase D — hardening and rollout

- Add tests, docs, and CI guards (skip CUDA tests when unavailable).
- Mark backend experimental initially; provide known limitations.
- Promote to stable after parity and perf thresholds are met.

## 5) Test plan additions

Add/extend tests near current backend coverage (`tests/test_raster_backend_config.py`, `tests/test_rasterizer.py`):

1. **Dispatch/config tests**
   - `VectorGraphicsScene.forward(...)` forwards `backend="cuda_gsplat"` correctly.
   - `fit_image(...)` stores/propagates CUDA backend config.
2. **Correctness tests** (CUDA-conditional)
   - Output shape/dtype/device invariants.
   - Gradient existence on key parameters (`control points`, `colors`, `opacities`, `_depth`).
   - Numerical agreement with `reference` within tolerance.
3. **Robustness tests**
   - Missing dependency -> clear `ImportError`/`RuntimeError` guidance.
   - Non-CUDA device request -> deterministic fallback or explicit failure (choose one policy and test it).

## 6) Performance acceptance criteria

Define explicit success criteria before merge-to-default:

- >= 2x speedup over `reference` on CUDA for 256x256 training step (forward+backward).
- Stable memory at target batch/curve counts (no unbounded growth across iterations).
- Image quality parity: PSNR/SSIM deltas within agreed tolerance compared to current backend for the same seed/config.

## 7) Risk register and mitigations

- **API mismatch with `gsplat` internals**
  - Mitigation: isolate adapter and keep custom CUDA fallback path.
- **Autograd instability in backward**
  - Mitigation: staged rollout with gradient checks and anomaly detection tests.
- **Depth/compositing semantic drift**
  - Mitigation: golden overlap scenes and explicit front-to-back unit tests.
- **Dependency friction in CI/dev setups**
  - Mitigation: optional extras + skip markers + clear install docs.

## 8) Proposed execution order (2-week practical sequence)

1. Week 1 (days 1-3): backend scaffolding + config propagation + failing placeholder tests.
2. Week 1 (days 4-5): `gsplat` adapter MVP + shape/grad tests passing on CUDA machine.
3. Week 2 (days 1-2): parity fixes + overlap/depth correctness tests.
4. Week 2 (days 3-4): benchmarking + tuning + docs.
5. Week 2 (day 5): stabilize, feature-flag decision, merge.

## 9) Minimal initial PR breakdown

- **PR1:** backend enum/plumbing + optional dependency + capability checks.
- **PR2:** adapter implementation + CUDA tests.
- **PR3:** benchmark updates + docs + rollout notes.

This sequence keeps risk contained and gives a clear fallback if `gsplat` integration does not meet parity/performance goals.
