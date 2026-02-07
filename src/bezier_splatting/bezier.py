"""Bézier curve primitives — pure PyTorch, fully batched."""

import torch
from jaxtyping import Float
from torch import Tensor

_BINOM_COEFFS: dict[int, list[int]] = {1: [1, 1], 2: [1, 2, 1], 3: [1, 3, 3, 1]}
_BINOM_TENSOR_CACHE: dict[tuple[int, str, int | None, torch.dtype], Tensor] = {}


def _cached_binom_tensor(degree: int, ref: Tensor) -> Tensor:
    """Return cached binomial coefficients tensor on ref's device/dtype."""
    key = (degree, ref.device.type, ref.device.index, ref.dtype)
    cached = _BINOM_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(_BINOM_COEFFS[degree], device=ref.device, dtype=ref.dtype)
        _BINOM_TENSOR_CACHE[key] = cached
    return cached


def bernstein_basis(t: Float[Tensor, " *batch"], degree: int) -> Float[Tensor, "*batch M1"]:
    """Evaluate Bernstein basis polynomials B_j^M(t).

    B_j^M(t) = C(M, j) * t^j * (1 - t)^(M - j)

    Args:
        t: Parameter values in [0, 1]. Shape: (*batch,)
        degree: Polynomial degree M.

    Returns:
        Bernstein basis values. Shape: (*batch, degree + 1)
    """
    if degree in _BINOM_COEFFS:
        binom = _cached_binom_tensor(degree, t)
    else:
        j = torch.arange(degree + 1, device=t.device, dtype=t.dtype)
        log_binom = (
            torch.lgamma(torch.tensor(degree + 1, device=t.device, dtype=t.dtype))
            - torch.lgamma(j + 1)
            - torch.lgamma(torch.tensor(degree, device=t.device, dtype=t.dtype) - j + 1)
        )
        binom = torch.exp(log_binom)  # (degree + 1,)

    j = torch.arange(degree + 1, device=t.device, dtype=t.dtype)
    t = t.unsqueeze(-1)  # (*batch, 1)
    # B_j^M(t) = C(M,j) * t^j * (1-t)^(M-j)
    basis = binom * t.pow(j) * (1 - t).pow(degree - j)  # (*batch, degree+1)
    return basis


def evaluate_bezier(control_points: Float[Tensor, "N CP 2"], t: Float[Tensor, " K"]) -> Float[Tensor, "N K 2"]:
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


def bezier_tangent(control_points: Float[Tensor, "N CP 2"], t: Float[Tensor, " K"]) -> Float[Tensor, "N K 2"]:
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


def evaluate_composite_bezier(
    control_points: Float[Tensor, "N 10 2"],
    num_samples: int,
    compute_tangents: bool = True,
) -> tuple[Float[Tensor, "N K 2"], Float[Tensor, "N K 2"] | None]:
    """Evaluate a composite Bézier curve (3 connected cubics sharing endpoints).

    The 10 control points are split into 3 segments:
        Segment 0: CPs [0, 1, 2, 3]
        Segment 1: CPs [3, 4, 5, 6]
        Segment 2: CPs [6, 7, 8, 9]

    Args:
        control_points: Shape (num_curves, 10, 2)
        num_samples: Total number of samples across all segments.
        compute_tangents: If False, skip tangent computation and return None.

    Returns:
        Tuple of (points, tangents). Points shape (num_curves, num_samples, 2).
        Tangents same shape, or None if compute_tangents is False.
    """
    # Distribute samples across 3 segments
    seg_sizes = composite_segment_sizes(num_samples)

    # Build per-segment control points: indices [0:4], [3:7], [6:10]
    seg_cps = torch.stack([
        control_points[:, 0:4, :],
        control_points[:, 3:7, :],
        control_points[:, 6:10, :],
    ], dim=1)  # (num_curves, 3, 4, 2)

    # Build a single t tensor of length K by concatenating per-segment linspaces.
    # Segment 0: linspace(0, 1, n0)
    # Segments 1, 2: linspace(0, 1, n+1)[1:] to skip shared endpoint at t=0
    t_parts = []
    seg_indices = []
    for seg_idx in range(3):
        n = seg_sizes[seg_idx]
        if n == 0:
            continue
        if seg_idx == 0:
            t_seg = torch.linspace(0, 1, n, device=control_points.device, dtype=control_points.dtype)
        else:
            t_seg = torch.linspace(0, 1, n + 1, device=control_points.device, dtype=control_points.dtype)[1:]
        t_parts.append(t_seg)
        seg_indices.append(torch.full((n,), seg_idx, device=control_points.device, dtype=torch.long))

    t_all = torch.cat(t_parts)          # (K,)
    seg_map = torch.cat(seg_indices)     # (K,) — which segment each sample belongs to

    # Gather CPs for each sample: seg_cps[:, seg_map] → (N, K, 4, 2)
    cp_per_sample = seg_cps[:, seg_map]  # (N, K, 4, 2)

    # Evaluate all samples in one batched call via Bernstein basis
    basis = bernstein_basis(t_all, 3)    # (K, 4)
    points = torch.einsum("sd,nsdx->nsx", basis, cp_per_sample)  # (N, K, 2)

    tangents = None
    if compute_tangents:
        delta_cp = cp_per_sample[:, :, 1:, :] - cp_per_sample[:, :, :-1, :]  # (N, K, 3, 2)
        basis_d = bernstein_basis(t_all, 2)  # (K, 3)
        tangents = 3 * torch.einsum("sd,nsdx->nsx", basis_d, delta_cp)  # (N, K, 2)

    return points, tangents
