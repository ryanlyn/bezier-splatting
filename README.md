# bezier-splatting

Unofficial educational reimplementation of [Bézier Splatting for Fast and Differentiable Vector Graphics Rendering](https://arxiv.org/abs/2503.16424).

This repository is for learning, inspection, and experimentation. It is not the official implementation and is not affiliated with the original authors.

## Attribution

All credit for the Bézier Splatting method belongs to the original researchers:

- Xi Liu
- Chaoyi Zhou
- Nanxuan Zhao
- Siyu Huang

Official resources:

- Paper: [arXiv 2503.16424](https://arxiv.org/abs/2503.16424)
- Official code: [xiliu8006/Bezier_splatting](https://github.com/xiliu8006/Bezier_splatting)
- Project page: [Bezier Splatting Project](https://xiliu8006.github.io/Bezier_splatting_project/)
- OpenReview entry: [NeurIPS 2025 poster](https://openreview.net/forum?id=bTclOYRfYJ)

## What Is Here

- Pure PyTorch Bézier-splatting renderer
- Optional CUDA raster backend via `gsplat`
- SVG export
- Reproduction and evaluation CLI
- Debug inspector and notebook
- Test suite covering geometry, rasterization, optimization, and eval plumbing

## Quickstart

```bash
uv sync --extra repro
uv run pytest tests/ -v --ignore=tests/test_reconstruction.py
uv run python -m eval.cli --help
```

Optional extras:

```bash
uv sync --extra debug
uv sync --extra cuda
```

## Sample Assets

The bundled sample PNGs in [samples/README.md](samples/README.md) come from the Kodak Lossless True Color Image Suite:

- Source: [r0k.us/graphics/kodak](https://r0k.us/graphics/kodak/)
- Included files: `kodim04.png`, `kodim07.png`, `kodim08.png`, `kodim23.png`

These files are included only as small educational examples for the debug tooling and local experiments.

## Notes

- This repo intentionally does not vendor the upstream paper PDF or source tree.
- Reproduction metadata should not be committed with machine-specific dataset paths.
- If you use this repository academically, cite the original Bézier Splatting paper rather than this reimplementation.

## Citation

```bibtex
@inproceedings{
  liu2025bzier,
  title={B\'ezier Splatting for Fast and Differentiable Vector Graphics Rendering},
  author={Xi Liu and Chaoyi Zhou and Nanxuan Zhao and Siyu Huang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=bTclOYRfYJ}
}
```
