"""CLI for paper/upstream reproduction pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .core import (
    DEFAULT_CURVE_COUNTS,
    DEFAULT_MODES,
    PAPER_CLOSED_STEPS,
    PAPER_OPEN_STEPS,
    benchmark_speed,
    build_manifest,
    collect_final_renders,
    evaluate_methods,
    generate_report,
    load_manifest,
    train_matrix,
)


def _csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _csv_strs(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _manifest_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("manifest", help="Build deterministic image manifest")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory with source images")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset label (default: basename of image-dir)")
    parser.add_argument("--output", type=Path, required=True, help="Manifest JSON output path")
    parser.add_argument("--subsample-every", type=int, default=1, help="Keep one out of every N images")
    parser.add_argument("--subsample-phase", type=int, default=0, help="Selection phase for subsampling")
    parser.add_argument("--upstream-div2k", action="store_true", help="Apply upstream DIV2K 1-in-4 selection (every=4, phase=3)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after subsampling")


def _train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Run matrix training from a manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True, help="Where per-image run artifacts are saved")
    parser.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES), help="Comma-separated modes")
    parser.add_argument("--curve-counts", type=str, default=",".join(str(v) for v in DEFAULT_CURVE_COUNTS))
    parser.add_argument("--train-device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip images that already have final outputs")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing per-image run dirs before retraining")
    parser.add_argument("--steps-open", type=int, default=PAPER_OPEN_STEPS)
    parser.add_argument("--steps-closed", type=int, default=PAPER_CLOSED_STEPS)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-svg", action="store_true")
    parser.add_argument("--raster-backend", type=str, default="auto")
    parser.add_argument("--raster-tile-size", type=int, default=16)
    parser.add_argument("--raster-chunk-size", type=int, default=None)


def _collect_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("collect", help="Copy final renders into upstream eval layout")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")


def _eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Evaluate methods against GT images")
    parser.add_argument("--source-root", type=Path, required=True, help="Root containing <method>/<dataset>/<image>.png")
    parser.add_argument("--target-dir", type=Path, required=True, help="Ground-truth image directory")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lpips-net", type=str, default="vgg")
    parser.add_argument("--no-ms-ssim", action="store_true")


def _report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("report", help="Generate consolidated report")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-ms-ssim", action="store_true")


def _speed_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("benchmark-speed", help="Benchmark forward/backward speed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--curves", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2040)
    parser.add_argument("--height", type=int, default=1344)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--raster-backend", type=str, default="auto")
    parser.add_argument("--raster-tile-size", type=int, default=16)
    parser.add_argument("--raster-chunk-size", type=int, default=None)


def _run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Run manifest + train + collect + eval + report")
    parser.add_argument("--image-dir", type=Path, required=True, help="Dataset image directory")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES))
    parser.add_argument("--curve-counts", type=str, default=",".join(str(v) for v in DEFAULT_CURVE_COUNTS))
    parser.add_argument("--train-device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--steps-open", type=int, default=PAPER_OPEN_STEPS)
    parser.add_argument("--steps-closed", type=int, default=PAPER_CLOSED_STEPS)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--raster-backend", type=str, default="auto")
    parser.add_argument("--raster-tile-size", type=int, default=16)
    parser.add_argument("--raster-chunk-size", type=int, default=None)
    parser.add_argument("--upstream-div2k", action="store_true", help="Use upstream DIV2K 1-in-4 selection")
    parser.add_argument("--subsample-every", type=int, default=1)
    parser.add_argument("--subsample-phase", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-ms-ssim", action="store_true")
    parser.add_argument("--run-speed-benchmark", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bezier Splatting paper/upstream reproduction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _manifest_parser(subparsers)
    _train_parser(subparsers)
    _collect_parser(subparsers)
    _eval_parser(subparsers)
    _report_parser(subparsers)
    _speed_parser(subparsers)
    _run_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "manifest":
        dataset_name = args.dataset_name or args.image_dir.name
        subsample_every = 4 if args.upstream_div2k else args.subsample_every
        subsample_phase = 3 if args.upstream_div2k else args.subsample_phase
        manifest = build_manifest(
            args.image_dir,
            args.output,
            dataset_name,
            subsample_every=subsample_every,
            subsample_phase=subsample_phase,
            limit=args.limit,
        )
        print(json.dumps({"manifest": str(args.output), "selected_count": manifest["selected_count"]}, indent=2))
        return

    if args.command == "train":
        manifest = load_manifest(args.manifest)
        summaries = train_matrix(
            manifest,
            args.runs_root,
            modes=_csv_strs(args.modes),
            curve_counts=_csv_ints(args.curve_counts),
            train_device=args.train_device,
            seed=args.seed,
            resume=args.resume,
            overwrite=args.overwrite,
            steps_open=args.steps_open,
            steps_closed=args.steps_closed,
            log_every=args.log_every,
            save_svg=args.save_svg,
            raster_backend=args.raster_backend,
            raster_tile_size=args.raster_tile_size,
            raster_chunk_size=args.raster_chunk_size,
        )
        print(json.dumps([asdict(summary) for summary in summaries], indent=2))
        return

    if args.command == "collect":
        counts = collect_final_renders(
            args.runs_root,
            args.eval_root,
            dataset_name=args.dataset_name,
            overwrite=args.overwrite,
        )
        print(json.dumps(counts, indent=2))
        return

    if args.command == "eval":
        results = evaluate_methods(
            args.source_root,
            args.target_dir,
            dataset_name=args.dataset_name,
            device=args.device,
            lpips_net=args.lpips_net,
            include_ms_ssim=not args.no_ms_ssim,
        )
        print(json.dumps(results, indent=2))
        return

    if args.command == "report":
        report = generate_report(
            args.eval_root,
            args.runs_root,
            args.out_dir,
            include_ms_ssim=not args.no_ms_ssim,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "benchmark-speed":
        out = benchmark_speed(
            args.output,
            device=args.device,
            curves=args.curves,
            width=args.width,
            height=args.height,
            warmup=args.warmup,
            iters=args.iters,
            raster_backend=args.raster_backend,
            raster_tile_size=args.raster_tile_size,
            raster_chunk_size=args.raster_chunk_size,
        )
        print(json.dumps(out, indent=2))
        return

    if args.command == "run":
        dataset_name = args.dataset_name or args.image_dir.name
        out_root: Path = args.out_root
        manifests_root = out_root / "manifests"
        runs_root = out_root / "runs"
        eval_root = out_root / "eval_inputs"
        report_root = out_root / "reports"

        manifest_path = manifests_root / f"{dataset_name}.json"
        subsample_every = 4 if args.upstream_div2k else args.subsample_every
        subsample_phase = 3 if args.upstream_div2k else args.subsample_phase

        manifest = build_manifest(
            args.image_dir,
            manifest_path,
            dataset_name,
            subsample_every=subsample_every,
            subsample_phase=subsample_phase,
            limit=args.limit,
        )

        summaries = train_matrix(
            manifest,
            runs_root,
            modes=_csv_strs(args.modes),
            curve_counts=_csv_ints(args.curve_counts),
            train_device=args.train_device,
            seed=args.seed,
            resume=args.resume,
            overwrite=args.overwrite,
            steps_open=args.steps_open,
            steps_closed=args.steps_closed,
            log_every=args.log_every,
            raster_backend=args.raster_backend,
            raster_tile_size=args.raster_tile_size,
            raster_chunk_size=args.raster_chunk_size,
        )

        collect_counts = collect_final_renders(
            runs_root,
            eval_root,
            dataset_name=dataset_name,
            overwrite=args.overwrite,
        )

        eval_results = evaluate_methods(
            eval_root,
            args.image_dir,
            dataset_name=dataset_name,
            device=args.train_device,
            include_ms_ssim=not args.no_ms_ssim,
        )

        report = generate_report(
            eval_root,
            runs_root,
            report_root,
            include_ms_ssim=not args.no_ms_ssim,
        )

        speed = None
        if args.run_speed_benchmark:
            speed = benchmark_speed(
                out_root / "speed" / "speed_benchmark.json",
                device=args.train_device,
                raster_backend=args.raster_backend,
                raster_tile_size=args.raster_tile_size,
                raster_chunk_size=args.raster_chunk_size,
            )

        payload = {
            "manifest": str(manifest_path),
            "summaries": [asdict(summary) for summary in summaries],
            "collect_counts": collect_counts,
            "eval_results": eval_results,
            "report": report,
            "speed": speed,
        }
        print(json.dumps(payload, indent=2))
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
