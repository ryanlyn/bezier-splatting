# MPS Rasterizer Backend

## Summary

The rasterizer now supports explicit backends:

- `reference`: original implementation (baseline correctness path)
- `mps`: vectorized tile path tuned to reduce Python-loop overhead
- `auto`: selects `mps` when the tensor device is `mps`, otherwise `reference`

Public entry point:

- `rasterize(..., backend="mps"|"auto"|"reference")`

Scene/training integration:

- `VectorGraphicsScene(..., raster_backend=..., raster_tile_size=..., raster_chunk_size=...)`
- `fit_image(..., raster_backend=..., raster_tile_size=..., raster_chunk_size=..., train_device=...)`

Reconstruction test device control:

- Set `BEZIER_TRAIN_DEVICE=mps` to force optimization on MPS.
- Set `BEZIER_TRAIN_DEVICE=auto` to use heuristic device selection on Apple Silicon.

## Tuning

Defaults:

- `backend="mps"`
- `tile_size=16`
- `chunk_size=16` for `reference`
- `chunk_size` for `mps` is auto-tuned when caller leaves default unchanged:
  - `96` for moderate Gaussian density (`G / n_tiles <= 20`)
  - `64` for denser scenes

Suggested workflow:

1. Run `python benchmarks/benchmark_rasterizer.py --device auto`.
2. Sweep `--chunk-size` for your hardware/workload.
3. Keep settings that maximize forward speed while preserving output parity.

## Validation

- `tests/test_rasterizer.py` keeps baseline rasterizer behavior checks.
- `tests/test_rasterizer_mps.py` verifies forward parity, gradient parity, and `auto` dispatch behavior.

## Benchmark command

```bash
python benchmarks/benchmark_rasterizer.py --device auto --sizes 64,128,256 --gaussians 1500,5000,15000
```

## Measured result (Apple MPS)

Command:

```bash
.venv/bin/python benchmarks/benchmark_rasterizer.py --device mps --sizes 64,128,256 --gaussians 1500,5000,15000 --tile-size 16 --chunk-size 64
```

Observed on this machine:

| Resolution | Gaussians | Ref Fwd (ms) | MPS Fwd (ms) | Fwd Speedup | Ref Bwd (ms) | MPS Bwd (ms) | Bwd Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1500 | 55.83 | 15.48 | 3.61x | 76.19 | 30.47 | 2.50x |
| 128 | 5000 | 210.85 | 32.29 | 6.53x | 276.20 | 71.24 | 3.88x |
| 256 | 15000 | 851.24 | 101.53 | 8.38x | 1042.55 | 220.70 | 4.72x |

Numerical parity check in benchmark: `max|reference - mps| <= 2.384e-07`.
