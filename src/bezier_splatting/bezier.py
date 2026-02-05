"""Bézier curve primitives — pure PyTorch, fully batched."""

from __future__ import annotations

import torch
from torch import Tensor


def bernstein_basis(t: Tensor, degree: int) -> Tensor:
    """Evaluate Bernstein basis polynomials B_j^M(t).

    B_j^M(t) = C(M, j) * t^j * (1 - t)^(M - j)

    Args:
        t: Parameter values in [0, 1]. Shape: (*batch,)
        degree: Polynomial degree M.

    Returns:
        Bernstein basis values. Shape: (*batch, degree + 1)
    """
    # Precompute binomial coefficients C(degree, j) for j = 0..degree
    j = torch.arange(degree + 1, device=t.device, dtype=t.dtype)
    # Use lgamma for numerical stability: C(M,j) = exp(lgamma(M+1) - lgamma(j+1) - lgamma(M-j+1))
    log_binom = (
        torch.lgamma(torch.tensor(degree + 1, device=t.device, dtype=t.dtype))
        - torch.lgamma(j + 1)
        - torch.lgamma(torch.tensor(degree, device=t.device, dtype=t.dtype) - j + 1)
    )
    binom = torch.exp(log_binom)  # (degree + 1,)

    t = t.unsqueeze(-1)  # (*batch, 1)
    # B_j^M(t) = C(M,j) * t^j * (1-t)^(M-j)
    basis = binom * t.pow(j) * (1 - t).pow(degree - j)  # (*batch, degree+1)
    return basis


def evaluate_bezier(control_points: Tensor, t: Tensor) -> Tensor:
    """Evaluate Bézier curves at parameter values t.

    Args:
        control_points: Control points. Shape: (num_curves, num_cp, 2)
        t: Parameter values in [0, 1]. Shape: (num_samples,)

    Returns:
        Points on curves. Shape: (num_curves, num_samples, 2)
    """
    degree = control_points.shape[1] - 1
    basis = bernstein_basis(t, degree)  # (num_samples, degree+1)
    # Einstein summation: curves × samples × control_points, control_points × dims
    points = torch.einsum("sd,cdx->csx", basis, control_points)
    return points


def bezier_tangent(control_points: Tensor, t: Tensor) -> Tensor:
    """Compute tangent vectors of Bézier curves at parameter values t.

    Uses the derivative identity: B'(t) = M * Σ B_j^{M-1}(t) * (P_{j+1} - P_j)

    Args:
        control_points: Control points. Shape: (num_curves, num_cp, 2)
        t: Parameter values in [0, 1]. Shape: (num_samples,)

    Returns:
        Tangent vectors. Shape: (num_curves, num_samples, 2)
    """
    degree = control_points.shape[1] - 1
    # Differences of consecutive control points
    delta_cp = control_points[:, 1:, :] - control_points[:, :-1, :]  # (num_curves, degree, 2)
    # Evaluate degree-(M-1) Bernstein basis
    basis = bernstein_basis(t, degree - 1)  # (num_samples, degree)
    # B'(t) = M * Σ B_j^{M-1}(t) * ΔP_j
    tangents = degree * torch.einsum("sd,cdx->csx", basis, delta_cp)
    return tangents


def composite_segment_sizes(num_samples: int) -> list[int]:
    """Return per-segment sample counts for a 3-segment composite curve.

    Extras go to earlier segments: e.g. 20 → [7, 7, 6].
    """
    base = num_samples // 3
    remainder = num_samples - 3 * base
    return [base + (1 if i < remainder else 0) for i in range(3)]


def evaluate_composite_bezier(control_points: Tensor, num_samples: int) -> tuple[Tensor, Tensor]:
    """Evaluate a composite Bézier curve (3 connected cubics sharing endpoints).

    The 10 control points are split into 3 segments:
        Segment 0: CPs [0, 1, 2, 3]
        Segment 1: CPs [3, 4, 5, 6]
        Segment 2: CPs [6, 7, 8, 9]

    Args:
        control_points: Shape (num_curves, 10, 2)
        num_samples: Total number of samples across all segments.

    Returns:
        Tuple of (points, tangents), each shape (num_curves, num_samples, 2).
        Points are ordered along the full curve from t=0 to t=1.
    """
    # Distribute samples across 3 segments
    seg_sizes = composite_segment_sizes(num_samples)

    # Build per-segment control points: indices [0:4], [3:7], [6:10]
    seg_cps = torch.stack([
        control_points[:, 0:4, :],
        control_points[:, 3:7, :],
        control_points[:, 6:10, :],
    ], dim=1)  # (num_curves, 3, 4, 2)

    all_points = []
    all_tangents = []
    for seg_idx in range(3):
        n = seg_sizes[seg_idx]
        if n == 0:
            continue
        # Avoid duplicating shared endpoints: skip t=0 for segments 1,2
        if seg_idx == 0:
            t = torch.linspace(0, 1, n, device=control_points.device, dtype=control_points.dtype)
        else:
            t = torch.linspace(0, 1, n + 1, device=control_points.device, dtype=control_points.dtype)[1:]

        cp = seg_cps[:, seg_idx]  # (num_curves, 4, 2)
        pts = evaluate_bezier(cp, t)  # (num_curves, n, 2)
        tng = bezier_tangent(cp, t)   # (num_curves, n, 2)
        all_points.append(pts)
        all_tangents.append(tng)

    points = torch.cat(all_points, dim=1)    # (num_curves, ~num_samples, 2)
    tangents = torch.cat(all_tangents, dim=1)
    return points, tangents
