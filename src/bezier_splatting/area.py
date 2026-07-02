"""Exact enclosed area computation for Bézier curves.

Uses Green's theorem to compute the exact enclosed area between paired
boundary curves: the signed area under a degree-n Bézier curve,
A = ∫₀¹ y(t) x'(t) dt, has a closed form that is bilinear in the control
points, obtained from the Bernstein product integral

    ∫₀¹ B_i^n(t) B_j^{n-1}(t) dt = C(n,i) C(n-1,j) / (2n · C(2n-1, i+j)).
"""

import math

import torch
from jaxtyping import Float
from torch import Tensor

_GREEN_COEFF_CACHE: dict[tuple[int, str, int | None, torch.dtype], Tensor] = {}


def _green_area_matrix(degree: int, ref: Tensor) -> Tensor:
    """Coefficient matrix M with A = Σᵢⱼ yᵢ M[i,j] Δxⱼ (exact ∫ y dx).

    M[i, j] = C(n,i) C(n-1,j) / (2 · C(2n-1, i+j)) for a degree-n curve.
    Cached per (degree, device, dtype).
    """
    key = (degree, ref.device.type, ref.device.index, ref.dtype)
    cached = _GREEN_COEFF_CACHE.get(key)
    if cached is None:
        n = degree
        m = torch.empty(n + 1, n, dtype=torch.float64)
        for i in range(n + 1):
            for j in range(n):
                m[i, j] = math.comb(n, i) * math.comb(n - 1, j) / (2.0 * math.comb(2 * n - 1, i + j))
        cached = m.to(device=ref.device, dtype=ref.dtype)
        _GREEN_COEFF_CACHE[key] = cached
    return cached


def bezier_signed_area(control_points: Float[Tensor, "*batch CP 2"]) -> Float[Tensor, " *batch"]:
    """Compute the exact signed area under a Bézier curve.

    The "signed area" is ∫₀¹ y(t) x'(t) dt — the area between the curve and
    the x-axis, evaluated exactly via the closed-form Bernstein product
    integral (Green's theorem). For a closed region formed by two curves
    sharing endpoints, the difference of their signed areas gives the
    enclosed area.

    Args:
        control_points: (..., num_cp, 2) tensor of control points.

    Returns:
        (...,) tensor of signed areas (can be negative depending on orientation).
    """
    cp = control_points
    degree = cp.shape[-2] - 1
    if degree < 1:
        return cp.new_zeros(cp.shape[:-2])

    x = cp[..., 0]  # (..., num_cp)
    y = cp[..., 1]  # (..., num_cp)
    dx = x[..., 1:] - x[..., :-1]  # (..., num_cp-1)

    coeff = _green_area_matrix(degree, cp)  # (num_cp, num_cp-1)
    return torch.einsum("...i,ij,...j->...", y, coeff, dx)


def closed_curve_enclosed_area(boundary_cp: Float[Tensor, "N 2 CP 2"]) -> Float[Tensor, " N"]:
    """Compute exact enclosed area between two paired boundary curves.

    The paired Bézier structure has two boundary curves that share start and
    end points, forming a closed region. The enclosed area is computed as the
    absolute difference of the signed areas of each boundary curve.

    Note: if the two boundaries cross each other (a "bowtie" region), the
    signed lobes partially cancel and the result underestimates the visual
    footprint of the shape.

    Args:
        boundary_cp: (N, 2, num_cp, 2) tensor where:
            - N = number of closed curves
            - 2 = two boundary curves
            - num_cp = control points per boundary
            - 2 = (x, y) coordinates

    Returns:
        (N,) tensor of positive enclosed areas.
    """
    if boundary_cp.shape[0] == 0:
        return torch.zeros(0, device=boundary_cp.device, dtype=boundary_cp.dtype)

    # Extract the two boundary curves
    boundary_0 = boundary_cp[:, 0]  # (N, num_cp, 2)
    boundary_1 = boundary_cp[:, 1]  # (N, num_cp, 2)

    # Compute signed area for each boundary
    area_0 = bezier_signed_area(boundary_0)  # (N,)
    area_1 = bezier_signed_area(boundary_1)  # (N,)

    # The enclosed area is the absolute difference
    # (boundaries go in opposite directions around the region)
    enclosed = torch.abs(area_0 - area_1)

    # Add small epsilon to avoid zero areas causing issues downstream
    enclosed = enclosed + 1e-6

    return enclosed
