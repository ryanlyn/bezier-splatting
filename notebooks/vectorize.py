"""Marimo notebook: Image → Vector Art via Bézier Splatting."""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Bézier Splatting: Image → Vector Graphics

        Upload a target image and optimize a set of differentiable Bézier curves
        to reconstruct it. The result is a resolution-independent SVG.

        Based on "Bézier Splatting for Fast and Differentiable Vector Graphics Rendering" (NeurIPS 2025).
        """
    )
    return


@app.cell
def _(mo):
    image_upload = mo.ui.file(
        filetypes=[".png", ".jpg", ".jpeg"],
        label="Upload target image",
    )
    image_upload
    return (image_upload,)


@app.cell
def _(mo):
    n_open_slider = mo.ui.slider(16, 512, value=128, step=16, label="Open curves")
    n_closed_slider = mo.ui.slider(0, 256, value=64, step=16, label="Closed curves")
    steps_slider = mo.ui.slider(1000, 30000, value=10000, step=1000, label="Optimization steps")
    resolution_slider = mo.ui.slider(64, 512, value=256, step=64, label="Resolution")

    mo.hstack([n_open_slider, n_closed_slider, steps_slider, resolution_slider])
    return n_open_slider, n_closed_slider, steps_slider, resolution_slider


@app.cell
def _(image_upload, mo, n_open_slider, n_closed_slider, steps_slider, resolution_slider):
    import io

    import torch
    from PIL import Image

    mo.stop(not image_upload.value, mo.md("*Upload an image to begin.*"))

    raw = image_upload.value[0].contents
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    res = resolution_slider.value
    img = img.resize((res, res), Image.Resampling.LANCZOS)

    # Convert to tensor (3, H, W) in [0, 1]
    import numpy as np
    target = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0

    config = {
        "n_open": n_open_slider.value,
        "n_closed": n_closed_slider.value,
        "steps": steps_slider.value,
        "H": res,
        "W": res,
    }

    mo.md(f"**Target loaded:** {res}×{res} | {config['n_open']} open + {config['n_closed']} closed curves | {config['steps']} steps")
    return target, config, img


@app.cell
def _(target, config, mo):
    run_button = mo.ui.run_button(label="Start Optimization")
    run_button
    return (run_button,)


@app.cell
def _(target, config, run_button, mo):
    mo.stop(not run_button.value, mo.md("*Click 'Start Optimization' to begin.*"))

    from bezier_splatting.optimization import fit_image
    from bezier_splatting.svg import scene_to_svg

    loss_log = []

    def log_callback(step, loss, scene):
        if step % 100 == 0:
            loss_log.append({"step": step, "loss": loss})

    scene = fit_image(
        target,
        n_open=config["n_open"],
        n_closed=config["n_closed"],
        steps=config["steps"],
        callback=log_callback,
    )

    rendered = scene(config["H"], config["W"]).detach()
    svg_str = scene_to_svg(scene, config["H"], config["W"])

    return scene, rendered, svg_str, loss_log


@app.cell
def _(target, rendered, mo, img):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(target.permute(1, 2, 0).numpy())
    axes[0].set_title("Target")
    axes[0].axis("off")

    axes[1].imshow(rendered.permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1].set_title("Rendered")
    axes[1].axis("off")

    error = (rendered - target).abs().mean(dim=0).numpy()
    axes[2].imshow(error, cmap="hot")
    axes[2].set_title("Error Map")
    axes[2].axis("off")

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(loss_log, mo):
    import matplotlib.pyplot as plt

    if loss_log:
        fig, ax = plt.subplots(figsize=(8, 4))
        steps = [entry["step"] for entry in loss_log]
        losses = [entry["loss"] for entry in loss_log]
        ax.plot(steps, losses)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Convergence")
        ax.set_yscale("log")
        mo.mpl.interactive(fig)
    return


@app.cell
def _(svg_str, mo):
    mo.md("### SVG Preview")
    mo.Html(svg_str)
    return


@app.cell
def _(svg_str, mo):
    mo.download(
        data=svg_str.encode("utf-8"),
        filename="vectorized.svg",
        mimetype="image/svg+xml",
        label="Download SVG",
    )
    return


if __name__ == "__main__":
    app.run()
