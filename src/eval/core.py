"""Paper/upstream reproduction utilities for Bezier Splatting.

This module implements a practical reproduction pipeline aligned with:
- Paper protocol in ``original_paper/arXiv-2503.16424v4``
- Upstream scripts in ``xiliu8006/Bezier_splatting``

It intentionally mirrors the upstream evaluation layout:
``<source_root>/<method>/<dataset_name>/<image_name>.png``
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from bezier_splatting.optimization import fit_image
from bezier_splatting.svg import scene_to_svg

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

PAPER_OPEN_STEPS = 15_000
PAPER_CLOSED_STEPS = 10_000
PAPER_PRUNE_EVERY = 400
PAPER_PRUNE_STOP_BEFORE_END = 1_000
PAPER_CP_LR = 2e-4
PAPER_SAMPLES_PER_OPEN = 20
PAPER_SAMPLES_PER_CLOSED_CURVE = 20
PAPER_NUM_INTERMEDIATE = 20

DEFAULT_CURVE_COUNTS = (256, 512, 1024)
DEFAULT_MODES = ("open", "closed")


@dataclass(slots=True)
class ManifestEntry:
    index: int
    filename: str
    stem: str
    path: str


@dataclass(slots=True)
class ExperimentRunSummary:
    experiment: str
    dataset_name: str
    mode: str
    curves: int
    images_total: int
    images_ran: int
    images_skipped: int
    total_training_seconds: float
    average_training_seconds: float


def _image_files(image_dir: Path) -> list[Path]:
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda p: p.name)
    return files


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _mean_ignore_nan(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return _mean(finite)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _to_tensor(path: Path, device: torch.device) -> Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return arr.to(device)


def _save_tensor_png(image_chw: Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (image_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(image_uint8).save(out_path)


def cp_lr_scale_for_resolution(height: int, width: int, cp_lr: float = PAPER_CP_LR) -> float:
    """Map absolute CP learning rate to this repo's resolution-scaled lr factor."""
    return cp_lr * max(height, width) / 0.25


def normalize_mode(mode: str) -> str:
    mode_norm = mode.strip().lower()
    if mode_norm in {"open", "unclosed"}:
        return "open"
    if mode_norm in {"closed", "area"}:
        return "closed"
    raise ValueError(f"Unknown mode: {mode!r}. Expected open/unclosed/closed/area.")


def experiment_name(mode: str, curves: int) -> str:
    """Default experiment naming matching upstream conventions."""
    mode_norm = normalize_mode(mode)
    if mode_norm == "open":
        return f"bezier_splatting_unclosed_our_{curves}"
    return f"bezier_splatting_area_our_{curves}"


def build_manifest(
    image_dir: Path,
    out_path: Path,
    dataset_name: str,
    *,
    subsample_every: int = 1,
    subsample_phase: int = 0,
    limit: int | None = None,
) -> dict:
    """Build a deterministic image manifest.

    Args:
        image_dir: Directory that directly contains image files.
        out_path: Output manifest JSON file.
        dataset_name: Dataset label for provenance.
        subsample_every: Keep one image every N images. Use 1 for no subsampling.
        subsample_phase: Zero-based phase in ``[0, N-1]``.
            Upstream DIV2K selection behavior is ``N=4, phase=3``.
        limit: Optional cap after subsampling.
    """
    files = _image_files(image_dir)

    if subsample_every < 1:
        raise ValueError("subsample_every must be >= 1")
    if not (0 <= subsample_phase < subsample_every):
        raise ValueError("subsample_phase must satisfy 0 <= phase < subsample_every")

    selected: list[Path]
    if subsample_every == 1:
        selected = files
    else:
        selected = [p for idx, p in enumerate(files) if idx % subsample_every == subsample_phase]

    if limit is not None:
        selected = selected[:limit]

    entries = [
        ManifestEntry(index=i, filename=p.name, stem=p.stem, path=str(p.relative_to(image_dir)))
        for i, p in enumerate(selected)
    ]

    manifest = {
        "dataset_name": dataset_name,
        "image_dir": str(image_dir),
        "total_discovered": len(files),
        "selected_count": len(entries),
        "selection": {
            "subsample_every": subsample_every,
            "subsample_phase": subsample_phase,
            "limit": limit,
        },
        "images": [
            {
                "index": entry.index,
                "filename": entry.filename,
                "stem": entry.stem,
                "path": entry.path,
            }
            for entry in entries
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(runs_root: Path, experiment: str, dataset_name: str, image_stem: str) -> Path:
    return runs_root / experiment / dataset_name / image_stem


def train_experiment(
    manifest: dict,
    runs_root: Path,
    *,
    mode: str,
    curves: int,
    train_device: str = "auto",
    seed: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    steps_open: int = PAPER_OPEN_STEPS,
    steps_closed: int = PAPER_CLOSED_STEPS,
    log_every: int = 200,
    save_svg: bool = False,
    optimizer_type: str = "adan",
    raster_backend: str = "auto",
    raster_tile_size: int = 16,
    raster_chunk_size: int | None = None,
) -> ExperimentRunSummary:
    """Train one (mode, curve_count) experiment across manifest images."""
    dataset_name = manifest["dataset_name"]
    images = manifest["images"]
    exp_name = experiment_name(mode, curves)
    mode_norm = normalize_mode(mode)
    device = _resolve_device(train_device)

    steps = steps_open if mode_norm == "open" else steps_closed
    n_open = curves if mode_norm == "open" else 0
    n_closed = curves if mode_norm == "closed" else 0

    ran = 0
    skipped = 0
    durations: list[float] = []

    for item in images:
        image_path = Path(item["path"])
        if not image_path.is_absolute():
            image_path = Path(manifest["image_dir"]) / image_path
        image_name = item["filename"]
        image_stem = item["stem"]
        out_dir = _run_dir(runs_root, exp_name, dataset_name, image_stem)
        out_png = out_dir / "final.png"
        out_meta = out_dir / "run.json"

        if overwrite and out_dir.exists():
            shutil.rmtree(out_dir)

        if resume and out_png.exists() and out_meta.exists():
            skipped += 1
            continue

        _set_seed(seed)
        target = _to_tensor(image_path, device)
        _, height, width = target.shape
        cp_lr_scale = cp_lr_scale_for_resolution(height, width)

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        scene = fit_image(
            target,
            n_open=n_open,
            n_closed=n_closed,
            steps=steps,
            prune_every=PAPER_PRUNE_EVERY,
            prune_stop_before_end=PAPER_PRUNE_STOP_BEFORE_END,
            # Scale only the control-point learning rate to the paper's
            # absolute value; other parameter groups keep their base rates.
            cp_lr_scale=cp_lr_scale,
            optimizer_type=optimizer_type,
            topology_schedule="alternating",
            topology_start_step=1000,
            topology_max_step_open=14_000,
            topology_max_step_closed=9_200,
            samples_per_open=PAPER_SAMPLES_PER_OPEN,
            samples_per_closed_curve=PAPER_SAMPLES_PER_CLOSED_CURVE,
            num_intermediate=PAPER_NUM_INTERMEDIATE,
            log_every=log_every,
            train_device=str(device),
            raster_backend=raster_backend,
            raster_tile_size=raster_tile_size,
            raster_chunk_size=raster_chunk_size,
        )
        elapsed = time.perf_counter() - t0

        rendered = scene(height, width).detach().cpu()
        _save_tensor_png(rendered, out_png)

        if save_svg:
            svg_text = scene_to_svg(scene)
            (out_dir / "final.svg").write_text(svg_text, encoding="utf-8")

        run_meta = {
            "experiment": exp_name,
            "mode": mode_norm,
            "curves": curves,
            "dataset_name": dataset_name,
            "image_name": image_name,
            "image_stem": image_stem,
            "image_path": str(image_path),
            "steps": steps,
            "height": height,
            "width": width,
            "training_seconds": elapsed,
            "cp_lr_scale": cp_lr_scale,
            "optimizer_type": optimizer_type,
            "paper_params": {
                "cp_lr_target": PAPER_CP_LR,
                "samples_per_open": PAPER_SAMPLES_PER_OPEN,
                "samples_per_closed_curve": PAPER_SAMPLES_PER_CLOSED_CURVE,
                "num_intermediate": PAPER_NUM_INTERMEDIATE,
                "prune_every": PAPER_PRUNE_EVERY,
                "prune_stop_before_end": PAPER_PRUNE_STOP_BEFORE_END,
            },
        }
        out_meta.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

        ran += 1
        durations.append(elapsed)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = ExperimentRunSummary(
        experiment=exp_name,
        dataset_name=dataset_name,
        mode=mode_norm,
        curves=curves,
        images_total=len(images),
        images_ran=ran,
        images_skipped=skipped,
        total_training_seconds=float(sum(durations)),
        average_training_seconds=_mean(durations) if durations else 0.0,
    )

    summary_path = runs_root / exp_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    return summary


def train_matrix(
    manifest: dict,
    runs_root: Path,
    *,
    modes: tuple[str, ...] = DEFAULT_MODES,
    curve_counts: tuple[int, ...] = DEFAULT_CURVE_COUNTS,
    train_device: str = "auto",
    seed: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    steps_open: int = PAPER_OPEN_STEPS,
    steps_closed: int = PAPER_CLOSED_STEPS,
    log_every: int = 200,
    save_svg: bool = False,
    optimizer_type: str = "adan",
    raster_backend: str = "auto",
    raster_tile_size: int = 16,
    raster_chunk_size: int | None = None,
) -> list[ExperimentRunSummary]:
    summaries: list[ExperimentRunSummary] = []
    for mode in modes:
        for curves in curve_counts:
            summary = train_experiment(
                manifest,
                runs_root,
                mode=mode,
                curves=curves,
                train_device=train_device,
                seed=seed,
                resume=resume,
                overwrite=overwrite,
                steps_open=steps_open,
                steps_closed=steps_closed,
                log_every=log_every,
                save_svg=save_svg,
                optimizer_type=optimizer_type,
                raster_backend=raster_backend,
                raster_tile_size=raster_tile_size,
                raster_chunk_size=raster_chunk_size,
            )
            summaries.append(summary)
    return summaries


def collect_final_renders(
    runs_root: Path,
    eval_root: Path,
    *,
    dataset_name: str,
    experiments: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Copy per-image ``final.png`` to upstream evaluation layout.

    Source layout:
      ``<runs_root>/<experiment>/<dataset_name>/<image_stem>/final.png``

    Destination layout:
      ``<eval_root>/<experiment>/<dataset_name>/<image_name>.png``
    """
    counts: dict[str, int] = {}

    exp_dirs = [p for p in runs_root.iterdir() if p.is_dir()] if runs_root.exists() else []
    exp_dirs.sort(key=lambda p: p.name)

    for exp_dir in exp_dirs:
        if experiments is not None and exp_dir.name not in experiments:
            continue

        dataset_dir = exp_dir / dataset_name
        if not dataset_dir.exists():
            continue

        copied = 0
        for image_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            src_png = image_dir / "final.png"
            meta_path = image_dir / "run.json"
            if not src_png.exists() or not meta_path.exists():
                continue

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            filename = meta.get("image_name", f"{image_dir.name}.png")
            dst = eval_root / exp_dir.name / dataset_name / filename
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists() and not overwrite:
                continue

            shutil.copy2(src_png, dst)
            copied += 1

        counts[exp_dir.name] = copied

    return counts


def _parse_method_identity(method_name: str) -> tuple[str, int] | None:
    pattern = re.compile(r"bezier_splatting_(?P<mode>area|closed|open|unclosed)_our_(?P<curves>\d+)$")
    match = pattern.match(method_name)
    if not match:
        return None
    mode_raw = match.group("mode")
    curves = int(match.group("curves"))
    mode = "open" if mode_raw in {"open", "unclosed"} else "closed"
    return mode, curves


def _load_lpips_model(device: torch.device, net: str = "vgg"):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "lpips is required for evaluation. Install with: uv add lpips"
        ) from exc

    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    return model


def _load_ssim_fns():
    try:
        from pytorch_msssim import ms_ssim, ssim
    except ImportError as exc:
        raise RuntimeError(
            "pytorch-msssim is required for evaluation. Install with: uv add pytorch-msssim"
        ) from exc
    return ssim, ms_ssim


def evaluate_methods(
    source_root: Path,
    target_dir: Path,
    *,
    dataset_name: str,
    device: str = "auto",
    lpips_net: str = "vgg",
    include_ms_ssim: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate all methods under ``source_root`` against GT in ``target_dir``."""
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    if not target_dir.exists():
        raise FileNotFoundError(f"target_dir does not exist: {target_dir}")

    torch_device = _resolve_device(device)
    lpips_model = _load_lpips_model(torch_device, net=lpips_net)
    ssim_fn, ms_ssim_fn = _load_ssim_fns()

    full_results: dict[str, dict[str, float]] = {}

    method_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for method_dir in method_dirs:
        rendered_dir = method_dir / dataset_name
        if not rendered_dir.exists():
            continue

        rendered_files = _image_files(rendered_dir)
        if not rendered_files:
            continue

        per_view: dict[str, dict[str, float]] = {
            "PSNR": {},
            "SSIM": {},
            "LPIPS": {},
        }
        if include_ms_ssim:
            per_view["MS_SSIM"] = {}

        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        lpips_vals: list[float] = []
        ms_ssim_vals: list[float] = []

        for rendered_path in rendered_files:
            target_path = target_dir / rendered_path.name
            if not target_path.exists():
                continue

            rendered = _to_tensor(rendered_path, torch_device).unsqueeze(0)
            target = _to_tensor(target_path, torch_device).unsqueeze(0)

            if rendered.shape != target.shape:
                continue

            with torch.no_grad():
                mse = F.mse_loss(rendered, target)
                psnr = float(10.0 * torch.log10(torch.tensor(1.0, device=torch_device) / mse).item())
                ssim_val = float(ssim_fn(rendered, target, data_range=1.0, size_average=True).item())
                lpips_val = float(lpips_model(rendered * 2.0 - 1.0, target * 2.0 - 1.0).mean().item())

                psnr_vals.append(psnr)
                ssim_vals.append(ssim_val)
                lpips_vals.append(lpips_val)

                per_view["PSNR"][rendered_path.name] = psnr
                per_view["SSIM"][rendered_path.name] = ssim_val
                per_view["LPIPS"][rendered_path.name] = lpips_val

                if include_ms_ssim:
                    try:
                        ms_val = float(ms_ssim_fn(rendered, target, data_range=1.0, size_average=True).item())
                    except RuntimeError:
                        ms_val = float("nan")
                    ms_ssim_vals.append(ms_val)
                    per_view["MS_SSIM"][rendered_path.name] = ms_val

        summary = {
            "PSNR": _mean_ignore_nan(psnr_vals),
            "SSIM": _mean_ignore_nan(ssim_vals),
            "LPIPS": _mean_ignore_nan(lpips_vals),
            "num_images": len(psnr_vals),
        }
        if include_ms_ssim:
            summary["MS_SSIM"] = _mean_ignore_nan(ms_ssim_vals)

        full_results[method_dir.name] = summary

        (source_root / f"{method_dir.name}_results.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (source_root / f"{method_dir.name}_per_view.json").write_text(
            json.dumps(per_view, indent=2),
            encoding="utf-8",
        )

    return full_results


def generate_report(
    eval_root: Path,
    runs_root: Path,
    out_dir: Path,
    *,
    include_ms_ssim: bool = True,
) -> dict:
    """Generate consolidated quantitative report from eval outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(eval_root.glob("*_results.json"), key=lambda p: p.name)
    rows: list[dict[str, object]] = []

    for result_file in result_files:
        method = result_file.name.removesuffix("_results.json")
        identity = _parse_method_identity(method)
        if identity is None:
            continue

        mode, curves = identity
        metrics = json.loads(result_file.read_text(encoding="utf-8"))

        summary_path = runs_root / method / "summary.json"
        if summary_path.exists():
            timing = json.loads(summary_path.read_text(encoding="utf-8"))
            avg_seconds = float(timing.get("average_training_seconds", float("nan")))
        else:
            avg_seconds = float("nan")

        row = {
            "method": method,
            "mode": "Open" if mode == "open" else "Closed",
            "curves": curves,
            "SSIM": float(metrics.get("SSIM", float("nan"))),
            "PSNR": float(metrics.get("PSNR", float("nan"))),
            "LPIPS": float(metrics.get("LPIPS", float("nan"))),
            "Opt_minutes": avg_seconds / 60.0 if math.isfinite(avg_seconds) else float("nan"),
            "num_images": int(metrics.get("num_images", 0)),
        }
        if include_ms_ssim:
            row["MS_SSIM"] = float(metrics.get("MS_SSIM", float("nan")))
        rows.append(row)

    rows.sort(key=lambda r: (str(r["mode"]), int(r["curves"])))

    csv_fields = ["method", "mode", "curves", "SSIM", "PSNR", "LPIPS", "Opt_minutes", "num_images"]
    if include_ms_ssim:
        csv_fields.insert(4, "MS_SSIM")

    csv_path = out_dir / "quantitative_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "## Context",
        "Quantitative reconstruction report aggregated from upstream-compatible evaluation outputs.",
        "",
        "## This PR",
        "| Mode | Curves | SSIM | MS-SSIM | PSNR | LPIPS | Opt. (min) | Images | Method |" if include_ms_ssim
        else "| Mode | Curves | SSIM | PSNR | LPIPS | Opt. (min) | Images | Method |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|" if include_ms_ssim
        else "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        if include_ms_ssim:
            md_lines.append(
                "| {mode} | {curves} | {SSIM:.4f} | {MS_SSIM:.4f} | {PSNR:.4f} | {LPIPS:.4f} | {Opt_minutes:.2f} | {num_images} | `{method}` |".format(
                    **row
                )
            )
        else:
            md_lines.append(
                "| {mode} | {curves} | {SSIM:.4f} | {PSNR:.4f} | {LPIPS:.4f} | {Opt_minutes:.2f} | {num_images} | `{method}` |".format(
                    **row
                )
            )

    md_path = out_dir / "quantitative_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    report = {
        "num_methods": len(rows),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "rows": rows,
    }
    (out_dir / "quantitative_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def benchmark_speed(
    out_path: Path,
    *,
    device: str = "auto",
    curves: int = 2048,
    width: int = 2040,
    height: int = 1344,
    warmup: int = 10,
    iters: int = 50,
    raster_backend: str = "auto",
    raster_tile_size: int = 16,
    raster_chunk_size: int | None = None,
) -> dict:
    """Benchmark forward/backward timings for open and closed scenes."""
    from bezier_splatting.model import VectorGraphicsScene

    torch_device = _resolve_device(device)

    def run_case(mode: str) -> dict[str, float]:
        n_open = curves if mode == "open" else 0
        n_closed = curves if mode == "closed" else 0
        scene = VectorGraphicsScene(
            n_open=n_open,
            n_closed=n_closed,
            H=height,
            W=width,
            samples_per_open=PAPER_SAMPLES_PER_OPEN,
            samples_per_closed_curve=PAPER_SAMPLES_PER_CLOSED_CURVE,
            num_intermediate=PAPER_NUM_INTERMEDIATE,
            raster_backend=raster_backend,
            raster_tile_size=raster_tile_size,
            raster_chunk_size=raster_chunk_size,
        ).to(torch_device)

        target = torch.rand(3, height, width, device=torch_device)
        forward_ms: list[float] = []
        backward_ms: list[float] = []

        total_steps = warmup + iters
        for step in range(total_steps):
            for p in scene.parameters():
                p.grad = None

            scene.update_depth_heuristic(height, width, update_open=True, update_closed=True)

            _synchronize(torch_device)
            t0 = time.perf_counter()
            rendered = scene(height, width)
            _synchronize(torch_device)
            t1 = time.perf_counter()

            loss = F.mse_loss(rendered, target)

            _synchronize(torch_device)
            t2 = time.perf_counter()
            loss.backward()
            _synchronize(torch_device)
            t3 = time.perf_counter()

            if step >= warmup:
                forward_ms.append((t1 - t0) * 1000.0)
                backward_ms.append((t3 - t2) * 1000.0)

        return {
            "forward_ms_mean": _mean(forward_ms),
            "forward_ms_std": float(torch.tensor(forward_ms).std(unbiased=False).item()) if forward_ms else float("nan"),
            "backward_ms_mean": _mean(backward_ms),
            "backward_ms_std": float(torch.tensor(backward_ms).std(unbiased=False).item()) if backward_ms else float("nan"),
            "iters": iters,
            "warmup": warmup,
        }

    open_stats = run_case("open")
    closed_stats = run_case("closed")

    out = {
        "resolution": {"width": width, "height": height},
        "curves": curves,
        "device": str(torch_device),
        "open": open_stats,
        "closed": closed_stats,
        "paper_reference": {
            "open_forward_ms": 4.5,
            "open_backward_ms": 4.7,
            "closed_forward_ms": 14.1,
            "closed_backward_ms": 24.58,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "## Context",
        "Speed benchmark for this repository under paper-like settings.",
        "",
        "## This PR",
        "| Mode | Forward (ms) | Backward (ms) | Iters | Device |",
        "|---|---:|---:|---:|---|",
        f"| Open | {open_stats['forward_ms_mean']:.3f} | {open_stats['backward_ms_mean']:.3f} | {iters} | `{torch_device}` |",
        f"| Closed | {closed_stats['forward_ms_mean']:.3f} | {closed_stats['backward_ms_mean']:.3f} | {iters} | `{torch_device}` |",
    ]
    out_path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return out
