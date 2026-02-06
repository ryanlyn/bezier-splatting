"""Micro-benchmark for rasterizer backends.

Measures forward and forward+backward latency for ``reference`` vs ``mps``
backends on the same device.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from dataclasses import dataclass

import torch

# Ensure ``python benchmarks/benchmark_rasterizer.py`` works without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bezier_splatting.rasterizer import rasterize
from bezier_splatting.sampling import GaussianParams


@dataclass
class BenchResult:
    forward_ms: float
    backward_ms: float
    max_abs_diff: float


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_gaussians(
    n_gaussians: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[GaussianParams, list[torch.Tensor]]:
    scale = torch.tensor([W, H], device=device, dtype=dtype)

    means = torch.rand(n_gaussians, 2, device=device, dtype=dtype) * scale
    scales = torch.rand(n_gaussians, 2, device=device, dtype=dtype) * 3.5 + 0.6
    rotations = torch.rand(n_gaussians, device=device, dtype=dtype) * (2.0 * math.pi)
    colors = torch.rand(n_gaussians, 3, device=device, dtype=dtype)
    opacities = torch.randn(n_gaussians, device=device, dtype=dtype)

    params = [means, scales, rotations, colors, opacities]
    if requires_grad:
        for p in params:
            p.requires_grad_(True)

    g = GaussianParams(
        means=means,
        scales=scales,
        rotations=rotations,
        colors=colors,
        opacities=opacities,
        curve_ids=torch.arange(n_gaussians, device=device, dtype=torch.long),
    )
    return g, params


def _time_forward(
    gaussians: GaussianParams,
    H: int,
    W: int,
    backend: str,
    tile_size: int,
    chunk_size: int,
    warmup: int,
    iters: int,
) -> float:
    device = gaussians.means.device

    for _ in range(warmup):
        _ = rasterize(
            gaussians,
            H,
            W,
            backend=backend,
            tile_size=tile_size,
            chunk_size=chunk_size,
        )
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = rasterize(
            gaussians,
            H,
            W,
            backend=backend,
            tile_size=tile_size,
            chunk_size=chunk_size,
        )
    _sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def _time_backward(
    n_gaussians: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
    backend: str,
    tile_size: int,
    chunk_size: int,
    warmup: int,
    iters: int,
) -> float:
    gaussians, params = _make_gaussians(
        n_gaussians,
        H,
        W,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    for _ in range(warmup):
        for p in params:
            p.grad = None
        out = rasterize(
            gaussians,
            H,
            W,
            backend=backend,
            tile_size=tile_size,
            chunk_size=chunk_size,
        )
        loss = out.mean()
        loss.backward()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        for p in params:
            p.grad = None
        out = rasterize(
            gaussians,
            H,
            W,
            backend=backend,
            tile_size=tile_size,
            chunk_size=chunk_size,
        )
        loss = out.mean()
        loss.backward()
    _sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def _run_case(
    H: int,
    W: int,
    n_gaussians: int,
    device: torch.device,
    dtype: torch.dtype,
    tile_size: int,
    chunk_size: int,
    warmup: int,
    iters: int,
) -> tuple[BenchResult, BenchResult]:
    g, _ = _make_gaussians(
        n_gaussians,
        H,
        W,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )

    ref_img = rasterize(g, H, W, backend="reference", tile_size=tile_size, chunk_size=chunk_size)
    mps_img = rasterize(g, H, W, backend="mps", tile_size=tile_size, chunk_size=chunk_size)
    max_abs = (ref_img - mps_img).abs().max().item()

    ref = BenchResult(
        forward_ms=_time_forward(g, H, W, "reference", tile_size, chunk_size, warmup, iters),
        backward_ms=_time_backward(n_gaussians, H, W, device, dtype, "reference", tile_size, chunk_size, warmup, iters),
        max_abs_diff=max_abs,
    )
    mps = BenchResult(
        forward_ms=_time_forward(g, H, W, "mps", tile_size, chunk_size, warmup, iters),
        backward_ms=_time_backward(n_gaussians, H, W, device, dtype, "mps", tile_size, chunk_size, warmup, iters),
        max_abs_diff=max_abs,
    )

    return ref, mps


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark rasterizer backends.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--sizes", default="64,128,256", help="Comma-separated square resolutions")
    parser.add_argument("--gaussians", default="1500,5000,15000", help="Comma-separated gaussian counts")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = torch.float32

    sizes = _parse_int_list(args.sizes)
    gaussians = _parse_int_list(args.gaussians)
    if len(sizes) != len(gaussians):
        raise ValueError("--sizes and --gaussians must have the same number of entries.")

    print(f"device={device} dtype={dtype} tile_size={args.tile_size} chunk_size={args.chunk_size}")
    print("")
    print("| Resolution | Gaussians | Ref Fwd (ms) | MPS Fwd (ms) | Fwd Speedup | Ref Bwd (ms) | MPS Bwd (ms) | Bwd Speedup | max|ref-mps| |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for size, n_g in zip(sizes, gaussians, strict=True):
        H = W = size
        ref, mps = _run_case(
            H,
            W,
            n_g,
            device=device,
            dtype=dtype,
            tile_size=args.tile_size,
            chunk_size=args.chunk_size,
            warmup=args.warmup,
            iters=args.iters,
        )
        fwd_speedup = ref.forward_ms / mps.forward_ms
        bwd_speedup = ref.backward_ms / mps.backward_ms
        print(
            "| "
            f"{size} | {n_g} | {ref.forward_ms:.2f} | {mps.forward_ms:.2f} | {fwd_speedup:.2f}x | "
            f"{ref.backward_ms:.2f} | {mps.backward_ms:.2f} | {bwd_speedup:.2f}x | {ref.max_abs_diff:.3e} |",
        )


if __name__ == "__main__":
    main()
