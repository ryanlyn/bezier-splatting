# Design: Circular Curve Initialization Pattern (Sec. 4.1)

## Context

Per the paper (Sec. 4.1):
> "Following the approach in LIVE, new curves are initialized in a circular pattern..."

The current implementation in `optimization.py` initializes new curves with **linear** patterns:
- `_make_open_curve`: spreads 10 CPs along x-axis with random noise
- `_make_closed_curve`: creates two horizontal boundary curves

This does not match the paper's stated approach.

## Problem Analysis

### Current Implementation Issues

**Open curves** (`_make_open_curve`, lines 453-471):
```python
t_vals = torch.linspace(-1, 1, 10, device=device)
cp[:, 0] = cx_n + t_vals * spread + torch.randn(...) * spread * 0.3
cp[:, 1] = cy_n + torch.randn(...) * spread * 0.5
```
This creates a horizontal line with noise—not circular.

**Closed curves** (`_make_closed_curve`, lines 474-502):
```python
bcp[b, :, 0] = cx_n + (t - 0.5) * size * 2  # x varies linearly
bcp[b, :, 1] = cy_n + y_off + noise         # y fixed offset + noise
```
This creates two horizontal curves forming a lens shape—not circular.

### Why Circular Initialization Matters

1. **Isotropy**: Circular initialization has no preferred direction, allowing optimization to deform curves toward features in any direction
2. **Stability**: Avoids directional bias that can cause curves to miss nearby features
3. **LIVE compatibility**: Matches the established approach from prior work

## Proposed Design

### Open Curves: Arc Initialization

Place 10 control points along a circular arc (not a full circle, since open curves have distinct endpoints).

```
Arc spans: θ ∈ [-π/2, π/2]  (half circle, 180°)
```

For each control point i ∈ [0, 9]:
```
θ_i = -π/2 + (π * i / 9)
x_i = cx + radius * cos(θ_i)
y_i = cy + radius * sin(θ_i)
```

This creates a smooth semicircular arc that:
- Has natural curvature (good for edge/stroke features)
- Can deform to lines, curves, or complex shapes
- Maintains continuity between the 3 cubic Bézier segments

### Closed Curves: Ellipse Initialization

For closed curves with 2 boundaries (each with `num_cp` control points), initialize as concentric ellipses:

**Outer boundary (b=0)**: Points along outer ellipse
**Inner boundary (b=1)**: Points along inner ellipse (smaller radius)

For each boundary b with radius `r_b`:
```
θ_i = π * i / (num_cp - 1)  # [0, π] for top arc
For b=0: place on top arc
For b=1: place on bottom arc (mirror)
```

Both boundaries share endpoints at θ=0 and θ=π, satisfying the shared endpoint constraint.

## Implementation Details

### `_make_open_curve` Changes

```python
def _make_open_curve(...) -> None:
    """Create one new open curve with circular arc initialization."""
    cx_n = cx_px / W
    cy_n = cy_px / H
    radius = 0.05  # same as current 'spread'

    cp = torch.zeros(10, 2, device=device)

    # Arc from -π/2 to π/2 (semicircle)
    theta = torch.linspace(-math.pi / 2, math.pi / 2, 10, device=device)
    cp[:, 0] = cx_n + radius * torch.cos(theta)
    cp[:, 1] = cy_n + radius * torch.sin(theta)

    # Small noise for variety
    cp += torch.randn_like(cp) * radius * 0.1
    cp = cp.clamp(0, 1)

    out_cps.append(cp)
    out_colors.append(color)
```

### `_make_closed_curve` Changes

```python
def _make_closed_curve(...) -> None:
    """Create one new closed curve with elliptical initialization."""
    cx_n = cx_px / W
    cy_n = cy_px / H

    # Outer and inner radii
    r_outer = 0.05
    r_inner = 0.03

    bcp = torch.zeros(2, num_cp, 2, device=device)
    theta = torch.linspace(0, math.pi, num_cp, device=device)

    # Boundary 0: top arc (outer radius)
    bcp[0, :, 0] = cx_n + r_outer * torch.cos(theta)
    bcp[0, :, 1] = cy_n + r_outer * torch.sin(theta)

    # Boundary 1: bottom arc (inner radius, mirrored)
    bcp[1, :, 0] = cx_n + r_inner * torch.cos(theta)
    bcp[1, :, 1] = cy_n - r_inner * torch.sin(theta)

    # Enforce shared endpoints (θ=0 and θ=π)
    shared_start = (bcp[0, 0] + bcp[1, 0]) / 2
    shared_end = (bcp[0, -1] + bcp[1, -1]) / 2
    bcp[0, 0] = shared_start
    bcp[1, 0] = shared_start
    bcp[0, -1] = shared_end
    bcp[1, -1] = shared_end

    # Small noise
    bcp[:, 1:-1] += torch.randn(2, num_cp - 2, 2, device=device) * r_outer * 0.1
    bcp = bcp.clamp(0, 1)

    out_cps.append(bcp)
    out_colors.append(color)
```

## Edge Cases

1. **Boundary clipping**: When center is near image edge, arc may extend outside [0,1]. Use `clamp(0, 1)` as current code does.

2. **Radius scaling**: The radius (0.05 normalized) matches current `spread` parameter. Could make this adaptive based on error region size.

3. **Shared endpoints**: The closed curve averaging logic must run after circular placement to ensure boundaries meet.

## Files to Modify

| File | Function | Change |
|------|----------|--------|
| `src/bezier_splatting/optimization.py` | `_make_open_curve` | Arc initialization |
| `src/bezier_splatting/optimization.py` | `_make_closed_curve` | Ellipse initialization |

## Testing Strategy

1. **Unit test**: Verify CPs lie on arc/ellipse within tolerance
2. **Visual test**: Render newly initialized curves, confirm circular appearance
3. **Regression test**: Run optimization on test image, verify metrics don't degrade

## Backward Compatibility

No API changes. The modification only affects internal initialization geometry. Existing code calling `fit_image()` will automatically use the new initialization.
