# Plan: Integrate gsplat CUDA backend

## Summary

Replace the rasterizer internals with gsplat's low-level CUDA primitives (`isect_tiles` + `isect_offset_encode` + `rasterize_to_pixels`). These accept pre-computed 2D means and conics directly — no 3D projection, no camera matrices, no quaternions. Merge the two existing Python backends into one. Total diff: ~+120 / -180 lines, almost entirely in `rasterizer.py`.

## Design decisions (from Opus agent debate)

### 1. Injection point: rasterizer-level only

Replace only `rasterize()` internals. The `model → GaussianParams → rasterizer → image` boundary stays exactly where it is. Sampling (Bezier evaluation, scale clamping, per-segment opacity, central-difference rotations) remains pure PyTorch — it's not the bottleneck and contains correctness-critical logic (mode-specific scale clamping that prevents NaN gradients).

### 2. Backend: gsplat v1.0+ low-level primitives

Use gsplat's public low-level API, NOT the high-level `rasterization()` wrapper:

```python
from gsplat import isect_tiles, isect_offset_encode, rasterize_to_pixels
```

These functions accept:
- `means2d: [G, 2]` — our pixel-space means directly
- `conics: [G, 3]` — upper triangle of Σ⁻¹, computed from our scales + rotations
- `colors: [G, 3]` — RGB values
- `opacities: [G]` — post-sigmoid values (we apply sigmoid before calling)
- `depths: [G]` — for per-tile sorting

No 3D projection. No quaternion conversion. No camera fabrication. No `eps2d` interference with our scale clamping.

**Why not the original paper's approach?** The original Bezier Splatting codebase uses a [custom fork of gsplat v0.1.x](https://github.com/XingtongGe/gsplat) with a hand-written CUDA kernel (`project_gaussians_2d_scale_rot`). That fork is pinned to a dead commit and requires building from source. Our approach uses standard gsplat (`pip install gsplat`) and computes conics in ~5 lines of PyTorch math instead.

**Why not a custom CUDA kernel?** A correct forward+backward 2D tile rasterizer is 500-1000 lines of CUDA + C++ bindings + build system changes. gsplat's primitives give the same performance with zero CUDA maintenance burden.

### 3. Fallback: one merged PyTorch backend

Merge `_rasterize_reference` and `_rasterize_mps` into a single `_rasterize_pytorch()`. The `mps` backend is strictly superior (vectorized gather, padded tile-aligned image) and already works on all devices (CPU, CUDA, MPS). The `reference` backend exists only for readability but has ~70% code overlap.

Result: 2 backends total — `pytorch` (fallback, works everywhere) and `gsplat` (CUDA-accelerated).

The PyTorch backend serves as ground truth for:
- Numerical parity regression tests (`torch.allclose(pytorch, gsplat, atol=1e-4)`)
- Gradient parity tests (autograd comparison)
- Mac/CPU fallback

### 4. Depth sorting: model keeps ownership

Model's `get_gaussians()` continues to own all depth logic (learned `_depth`, heuristic overwrites, boundary `-1e-6` offset, global argsort). Gaussians arrive at the rasterizer pre-sorted.

For gsplat's `isect_tiles`, pass `depths = torch.arange(G, dtype=float32)` as synthetic monotonic depths. Since Gaussians are already globally sorted, monotonic indices guarantee gsplat's per-tile radix sort preserves our order. No fake z-coordinates, no float precision concerns.

### 5. Opacity: apply sigmoid before gsplat

gsplat expects post-sigmoid opacities in [0, 1]. Our system stores pre-sigmoid. The adapter applies `torch.sigmoid()` before the gsplat call. The gradient flows through this correctly.

## File changes

### `src/bezier_splatting/rasterizer.py` (primary change)

1. **Delete** `_rasterize_reference()` (~180 lines removed)
2. **Rename** `_rasterize_mps()` → `_rasterize_pytorch()`
3. **Add** `_rasterize_gsplat()` (~60 lines):
   ```python
   def _rasterize_gsplat(gaussians, H, W, bg_color=None, tile_size=16):
       # 1. Apply sigmoid to opacities (gsplat expects post-sigmoid)
       opacities = torch.sigmoid(gaussians.opacities)

       # 2. Build conics from scales + rotations (~5 lines)
       #    Σ = R·diag(σ²)·Rᵀ → Σ⁻¹ → upper triangle [a, b, d]
       cov = _build_covariance(gaussians.scales, gaussians.rotations)
       inv_cov, _ = _invert_2x2(cov)
       conics = torch.stack([inv_cov[:, 0, 0], inv_cov[:, 0, 1], inv_cov[:, 1, 1]], dim=-1)

       # 3. Compute radii (3σ bounding)
       radii = (3.0 * torch.sqrt(torch.stack([cov[:, 0, 0], cov[:, 1, 1]], dim=-1))).max(dim=-1).values.int()

       # 4. Synthetic monotonic depths (Gaussians already globally sorted)
       depths = torch.arange(G, device=means.device, dtype=torch.float32)

       # 5. Tile intersection
       tile_width = (W + tile_size - 1) // tile_size
       tile_height = (H + tile_size - 1) // tile_size
       _, isect_ids, flatten_ids = isect_tiles(means, radii, depths, tile_size, tile_width, tile_height)
       isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

       # 6. Rasterize
       rendered, _ = rasterize_to_pixels(means, conics, colors, opacities, H, W, tile_size, isect_offsets, flatten_ids, backgrounds=bg_color)
       return rendered.squeeze(0).permute(2, 0, 1)  # (H,W,3) → (3,H,W)
   ```
4. **Update** `_resolve_backend()`: support `"gsplat"`, `"pytorch"`, `"auto"`. Keep `"mps"` and `"reference"` as aliases for `"pytorch"` (backward compat in configs/tests).
5. **Update** `rasterize()` dispatch.
6. **Update** `RasterBackend` type alias.

### `pyproject.toml`

Add optional CUDA dependency:
```toml
[project.optional-dependencies]
cuda = ["gsplat>=1.0"]
```

### `src/bezier_splatting/optimization.py`

Update `fit_image()` docstring for `raster_backend` param to document `"gsplat"` option. Update `_resolve_auto_device()` to prefer gsplat on CUDA.

### Tests

- **Update `test_rasterizer_mps.py`** → rename to `test_rasterizer_parity.py`. Test pytorch backend against itself (replaces reference vs mps tests). Add CUDA-conditional gsplat parity tests.
- **Update `test_raster_backend_config.py`** → update backend string references.
- **Update `test_rasterizer.py`** → keep as-is (tests core math + rendering behavior, backend-agnostic).
- **Add gsplat-specific tests** (CUDA-conditional):
  - Output shape/dtype/device invariants
  - Numerical parity vs pytorch backend (atol=1e-4)
  - Gradient existence on means, scales, rotations, colors, opacities
  - Empty scene handling
  - Front-to-back compositing order preservation
  - Missing dependency → clear error message

## Implementation tasks

1. **Merge Python backends** — delete `_rasterize_reference`, rename `_rasterize_mps` → `_rasterize_pytorch`, update `_resolve_backend` and `rasterize` dispatch. Update all tests.
2. **Add gsplat backend** — implement `_rasterize_gsplat`, add to dispatch, add optional dependency, add lazy import with clear error.
3. **Add gsplat tests** — CUDA-conditional parity tests, gradient tests, edge cases.
4. **Update optimization.py** — auto-device logic, docstrings.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| gsplat opacity is post-sigmoid, ours is pre | Apply sigmoid in adapter; verified in tests |
| Compositing order mismatch | Parity test on overlapping Gaussians; synthetic depths preserve our sort |
| gsplat breaking changes | Pin `gsplat>=1.0,<2.0` |
| Conics format mismatch | Verify with single-Gaussian rendering parity test |
| Missing gsplat on Mac/CI | Lazy import + `pytest.mark.skipif` + clear `RuntimeError` |

## Performance expectations

- Current: ~205ms forward+backward at 256x256, 500 Gaussians
- Expected with gsplat: ~5-20ms (10-40x speedup)
- Target: ≥2x (easily exceeded)
