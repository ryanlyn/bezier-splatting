# Design: Depth Assignment Using True Enclosed Area

## Overview

Replace the current bounding-box approximation for closed curve area calculation with the mathematically exact enclosed area. This improves depth ordering accuracy for diagonal/curved shapes.

## Current Implementation

**Location:** `src/bezier_splatting/model.py` lines 174-178

```python
# Per-curve area from bounding box of CPs
bcp_px = bcp * torch.tensor([W, H], device=bcp.device, dtype=bcp.dtype)
cp_flat = bcp_px.reshape(bcp_px.shape[0], -1, 2)
bb = cp_flat.max(dim=1).values - cp_flat.min(dim=1).values
areas = bb[:, 0] * bb[:, 1]  # Bounding box area
```

**Problem:** A diagonal ellipse might have a bounding box 2-3x larger than its true area, causing incorrect depth ordering relative to other shapes.

## Proposed Solution

### Mathematical Background

For a closed region bounded by two Bézier curves sharing endpoints (the "paired Bézier curve structure" from the paper), the true enclosed area can be computed using the **shoelace formula** applied to the Bézier curves.

For a parametric curve C(t) = (x(t), y(t)), the signed area under the curve is:

```
A = ∫₀¹ x(t) · y'(t) dt
```

For a cubic Bézier curve with control points P₀, P₁, P₂, P₃, this has a closed-form solution:

```
A = 3/20 * [(x₀ + x₁)(y₁ - y₀) + (x₁ + x₂)(y₂ - y₁) + (x₂ + x₃)(y₃ - y₂) +
           (x₃ + x₀)(y₀ - y₃) + 2*(x₁*y₂ - x₂*y₁)]
```

For the paired structure with boundary curves B₁ and B₂, the enclosed area is:

```
enclosed_area = |area_under_B1 - area_under_B2|
```

### Implementation Plan

1. **New utility function** in `src/bezier_splatting/area.py`:
   ```python
   def bezier_signed_area(control_points: Tensor) -> Tensor:
       """
       Compute signed area under a Bézier curve using Green's theorem.

       Args:
           control_points: (N, num_cp, 2) tensor of control points

       Returns:
           (N,) tensor of signed areas
       """
   ```

2. **Closed curve area function**:
   ```python
   def closed_curve_enclosed_area(
       boundary_cp_1: Tensor,  # (N, num_cp, 2) first boundary
       boundary_cp_2: Tensor,  # (N, num_cp, 2) second boundary
   ) -> Tensor:
       """
       Compute true enclosed area between two boundary curves.

       Returns:
           (N,) tensor of positive enclosed areas
       """
   ```

3. **Integration in model.py**:
   - Replace bounding box calculation with `closed_curve_enclosed_area()`
   - Keep pixel-space scaling for consistency with open curve areas

### Edge Cases

1. **Self-intersecting curves**: The shoelace formula gives signed area, so self-intersecting regions may partially cancel. Use `abs()` and accept this limitation (rare in practice due to Xing loss).

2. **Degenerate curves**: If boundary curves are identical, area = 0. Handle by adding small epsilon to avoid division issues downstream.

3. **Very small areas**: The pruning threshold (area > 4.0 in optimization.py) needs no change since we're computing in pixel space.

## Alternatives Considered

1. **Monte Carlo sampling**: Sample random points, check if inside region. Rejected: slow, noisy gradients.

2. **Triangulation**: Tessellate region and sum triangle areas. Rejected: more complex, not differentiable.

3. **Numerical integration**: Approximate ∫ y dx numerically. Rejected: closed-form exists and is faster.

## Testing Strategy

1. Unit tests comparing against known geometric areas (circle, rectangle at various rotations)
2. Integration tests verifying depth ordering behavior
3. Visual regression tests for existing vectorization results

## Files Modified

- `src/bezier_splatting/area.py` (new)
- `src/bezier_splatting/model.py` (update closed curve area calculation)
- `src/bezier_splatting/optimization.py` (update pruning area calculation)
- `src/bezier_splatting/svg.py` (update SVG export sorting)
- `tests/test_area.py` (new)
