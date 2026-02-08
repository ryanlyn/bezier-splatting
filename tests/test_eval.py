from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from eval.core import (
    _parse_method_identity,
    build_manifest,
    collect_final_renders,
    experiment_name,
)


def _make_dummy_png(path: Path, value: int) -> None:
    arr = torch.full((8, 8, 3), value, dtype=torch.uint8).numpy()
    Image.fromarray(arr).save(path)


def test_build_manifest_matches_upstream_one_in_four(tmp_path: Path) -> None:
    image_dir = tmp_path / "DIV2K_HR"
    image_dir.mkdir(parents=True)

    for i in range(1, 9):
        _make_dummy_png(image_dir / f"{i:05}.png", value=i)

    out_path = tmp_path / "manifest.json"
    manifest = build_manifest(
        image_dir,
        out_path,
        "DIV2K_HR",
        subsample_every=4,
        subsample_phase=3,
    )

    assert manifest["selected_count"] == 2
    names = [entry["filename"] for entry in manifest["images"]]
    assert names == ["00004.png", "00008.png"]


def test_collect_final_renders_copies_to_eval_layout(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    eval_root = tmp_path / "eval"
    dataset_name = "DIV2K_HR"
    exp_name = experiment_name("closed", 256)

    run_dir = runs_root / exp_name / dataset_name / "00004"
    run_dir.mkdir(parents=True)

    _make_dummy_png(run_dir / "final.png", value=123)
    (run_dir / "run.json").write_text(
        json.dumps({"image_name": "00004.png"}),
        encoding="utf-8",
    )

    counts = collect_final_renders(runs_root, eval_root, dataset_name=dataset_name)

    assert counts[exp_name] == 1
    assert (eval_root / exp_name / dataset_name / "00004.png").exists()


def test_parse_method_identity_handles_upstream_names() -> None:
    assert _parse_method_identity("bezier_splatting_unclosed_our_512") == ("open", 512)
    assert _parse_method_identity("bezier_splatting_area_our_1024") == ("closed", 1024)
    assert _parse_method_identity("unknown_method") is None
