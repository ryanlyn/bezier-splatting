"""Image quality metrics for reconstruction evaluation."""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def compute_mse(rendered: Float[Tensor, "C H W"], target: Float[Tensor, "C H W"]) -> Float[Tensor, ""]:
    """Mean squared error between rendered and target images."""
    return F.mse_loss(rendered, target)


def compute_psnr(rendered: Float[Tensor, "C H W"], target: Float[Tensor, "C H W"]) -> Float[Tensor, ""]:
    """Peak Signal-to-Noise Ratio in dB. Higher is better."""
    mse = compute_mse(rendered, target)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(1.0 / mse)


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> Float[Tensor, " K"]:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def compute_ssim(rendered: Float[Tensor, "C H W"], target: Float[Tensor, "C H W"], window_size: int = 11) -> Float[Tensor, ""]:
    """Structural Similarity Index. Higher is better (max 1.0).

    Args:
        rendered: (3, H, W) or (1, H, W) in [0, 1]
        target: same shape as rendered

    Returns:
        Scalar SSIM value.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Ensure (C, H, W) → (1, C, H, W)
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        target = target.unsqueeze(0)

    channels = rendered.shape[1]

    # Create Gaussian window
    kernel_1d = _gaussian_kernel_1d(window_size, 1.5, rendered.device)
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (ws, ws)
    window = kernel_2d.expand(channels, 1, window_size, window_size)

    pad = window_size // 2

    mu1 = F.conv2d(rendered, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(rendered ** 2, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(rendered * target, window, padding=pad, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


def _sobel_edges(image: Float[Tensor, "C H W"]) -> Float[Tensor, "1 H W"]:
    """Detect edges using Sobel operator. Returns edge magnitude (1, H, W)."""
    # Convert to grayscale
    if image.shape[0] == 3:
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    else:
        gray = image[0]
    gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gray.dtype, device=gray.device).reshape(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)

    return torch.sqrt(gx ** 2 + gy ** 2).squeeze(0)  # (1, H, W)


def compute_edge_mse(rendered: Float[Tensor, "C H W"], target: Float[Tensor, "C H W"]) -> Float[Tensor, ""]:
    """MSE computed only at Sobel-detected edge pixels of the target.

    Exposes boundary reconstruction failures.
    """
    edges = _sobel_edges(target)  # (1, H, W)
    # Threshold to get edge mask
    threshold = edges.mean() + edges.std()
    edge_mask = (edges > threshold).float()  # (1, H, W)

    if edge_mask.sum() == 0:
        return compute_mse(rendered, target)

    # Expand mask to match channels
    mask = edge_mask.expand_as(rendered)
    masked_diff = (rendered - target) ** 2 * mask
    return masked_diff.sum() / mask.sum()


def compute_metrics(rendered: Float[Tensor, "C H W"], target: Float[Tensor, "C H W"]) -> dict[str, float]:
    """Compute all metrics between rendered and target images.

    Args:
        rendered: (3, H, W) in [0, 1]
        target: (3, H, W) in [0, 1]

    Returns:
        Dict with keys: mse, psnr, ssim, edge_mse, per_channel_mse
    """
    rendered = rendered.detach().clamp(0, 1)
    target = target.detach()

    per_ch = [(rendered[c] - target[c]).pow(2).mean().item() for c in range(3)]

    return {
        "mse": compute_mse(rendered, target).item(),
        "psnr": compute_psnr(rendered, target).item(),
        "ssim": compute_ssim(rendered, target).item(),
        "edge_mse": compute_edge_mse(rendered, target).item(),
        "per_channel_mse": per_ch,
    }
