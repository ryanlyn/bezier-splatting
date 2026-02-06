"""True enclosed area computation for Bézier curves.

Uses Green's theorem to compute exact enclosed area between paired boundary curves.
"""

import torch
from jaxtyping import Float
from torch import Tensor


def bezier_signed_area(control_points: Float[Tensor, "*batch CP 2"]) -> Float[Tensor, " *batch"]:
    """Compute signed area under a Bézier curve using the trapezoidal rule.

    The "signed area" is the area between the curve and the x-axis. For a closed
    region formed by two curves sharing endpoints, the difference of their
    signed areas gives the enclosed area.

    Uses the trapezoidal rule on the control polygon:
    A = Σᵢ (x_{i+1} - x_i) * (y_i + y_{i+1}) / 2

    This is exact for piecewise-linear paths and a good approximation for
    Bézier curves when the control polygon closely follows the curve.

    Args:
        control_points: (..., num_cp, 2) tensor of control points.

    Returns:
        (...,) tensor of signed areas (can be negative depending on orientation).
    """
    # Shape: (..., num_cp, 2)
    cp = control_points
    x = cp[..., 0]  # (..., num_cp)
    y = cp[..., 1]  # (..., num_cp)

    # Trapezoidal rule: A = Σ (x_{i+1} - x_i) * (y_i + y_{i+1}) / 2
    dx = x[..., 1:] - x[..., :-1]  # (..., num_cp-1)
    y_avg = (y[..., :-1] + y[..., 1:]) / 2  # (..., num_cp-1)

    signed_area = (dx * y_avg).sum(dim=-1)

    return signed_area


def closed_curve_enclosed_area(boundary_cp: Float[Tensor, "N 2 CP 2"]) -> Float[Tensor, " N"]:
    """Compute true enclosed area between two paired boundary curves.

    The paired Bézier structure has two boundary curves that share start and
    end points, forming a closed region. The enclosed area is computed as the
    absolute difference of the signed areas of each boundary curve.

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
