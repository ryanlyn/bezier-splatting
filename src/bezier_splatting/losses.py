"""Configurable composite loss system for Bezier Splatting optimization.

Implements reconstruction losses, regularizers from the original paper, and
the LIVE Xing loss for self-intersection prevention. All loss terms can be
individually enabled/disabled and weighted via LossConfig.

All control points are in [-1, 1] model space.
Colors and opacities are stored in pre-sigmoid space.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .bezier import evaluate_bezier
from .metrics import compute_ssim
from .model import VectorGraphicsScene


@dataclass
class LossConfig:
    """Configuration for which loss terms to apply and their weights."""

    loss_type: str = "L2"  # "L2", "L1", "Fusion1" (0.7*MSE + 0.3*(1-SSIM))
    lambda_xing: float = 0.0  # disabled by default (original has it commented out)
    lambda_shape: float = 1e-2  # shape regularizer weight
    lambda_opacity_prior: float = 1e-2  # opacity prior weight
    lambda_curvature: float = 1.0  # curvature regularizer weight
    lambda_boundary: float = 1.0  # boundary joint constraint weight
    apply_shape_reg: bool = True  # closed curves only
    apply_opacity_prior: bool = True  # closed curves only
    apply_curvature: bool = True  # closed curves only
    apply_boundary: bool = True  # both curve types


# ── Reconstruction Losses ────────────────────────────────────────────────


def reconstruction_loss(
    rendered: Float[Tensor, "C H W"],
    target: Float[Tensor, "C H W"],
    loss_type: str = "L2",
    lambda_value: float = 0.7,
) -> Float[Tensor, ""]:
    """Compute reconstruction loss between rendered and target images.

    Args:
        rendered: Rendered image (3, H, W) in [0, 1].
        target: Target image (3, H, W) in [0, 1].
        loss_type: One of "L2", "L1", or "Fusion1".
        lambda_value: Blend factor for Fusion1 (lambda*MSE + (1-lambda)*(1-SSIM)).

    Returns:
        Scalar loss value.
    """
    if loss_type == "L2":
        return F.mse_loss(rendered, target)
    elif loss_type == "L1":
        return F.l1_loss(rendered, target)
    elif loss_type == "Fusion1":
        mse = F.mse_loss(rendered, target)
        ssim = compute_ssim(rendered, target)
        return lambda_value * mse + (1 - lambda_value) * (1 - ssim)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Expected 'L2', 'L1', or 'Fusion1'.")


# ── Regularizers ─────────────────────────────────────────────────────────


def shape_regularizer(
    control_points: Float[Tensor, "N CP 2"],
    degree: int = 3,
) -> Float[Tensor, ""]:
    """Penalize interior CPs that project outside the [0, 1] span of the chord.

    For each curve, projects interior control points onto the line from P0 to
    P_end and penalizes projection parameters alpha < 0 or alpha > 1.

    Args:
        control_points: (N, CP, 2) control points in any coordinate space.
        degree: Bezier degree (used for documentation; all interior CPs are penalized).

    Returns:
        Scalar loss value.
    """
    n = control_points.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=control_points.device)

    p0 = control_points[:, 0, :]  # (N, 2)
    p_end = control_points[:, -1, :]  # (N, 2)
    interior = control_points[:, 1:-1, :]  # (N, CP-2, 2)

    if interior.shape[1] == 0:
        return torch.tensor(0.0, device=control_points.device)

    v = p_end - p0  # (N, 2) chord vector
    v_sq = (v * v).sum(dim=-1, keepdim=True).unsqueeze(1)  # (N, 1, 1)
    v_sq = v_sq.clamp(min=1e-12)

    u = interior - p0.unsqueeze(1)  # (N, CP-2, 2)
    # alpha = dot(u, v) / |v|^2
    alpha = (u * v.unsqueeze(1)).sum(dim=-1, keepdim=True) / v_sq  # (N, CP-2, 1)
    alpha = alpha.squeeze(-1)  # (N, CP-2)

    loss = (F.relu(alpha - 1).pow(2) + F.relu(-alpha).pow(2)).sum() / n
    return loss


def opacity_prior(
    opacities: Float[Tensor, " N"],
) -> Float[Tensor, ""]:
    """Push opacities toward full visibility.

    Penalizes deviation of sigmoid(opacity) from 1.0.

    Args:
        opacities: (N,) pre-sigmoid opacity values.

    Returns:
        Scalar loss value.
    """
    if opacities.numel() == 0:
        return torch.tensor(0.0, device=opacities.device)

    return (torch.sigmoid(opacities) - 1.0).abs().mean()


def curvature_loss(
    boundary_cp: Float[Tensor, "N 2 CP 2"],
    H: int,
    W: int,
    samples_per_boundary: int = 30,
) -> Float[Tensor, ""]:
    """Penalize high curvature in closed curve boundaries.

    For each closed curve, samples points along both boundaries, concatenates
    them (second boundary flipped) to form a closed loop, then penalizes
    second-order finite differences where the angle is below 60 degrees.

    Args:
        boundary_cp: (N, 2, CP, 2) boundary control points in [-1, 1].
        H: Image height (for pixel-space scaling of curvature).
        W: Image width.
        samples_per_boundary: Number of sample points per boundary curve.

    Returns:
        Scalar loss value.
    """
    n = boundary_cp.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=boundary_cp.device)

    num_cp = boundary_cp.shape[2]
    degree = num_cp - 1

    # Sample points along both boundaries
    t = torch.linspace(0, 1, samples_per_boundary, device=boundary_cp.device, dtype=boundary_cp.dtype)

    # Boundary 0: forward direction, Boundary 1: reversed
    b0_cp = boundary_cp[:, 0, :, :]  # (N, CP, 2)
    b1_cp = boundary_cp[:, 1, :, :]  # (N, CP, 2)

    b0_pts = evaluate_bezier(b0_cp, t)  # (N, K, 2)
    b1_pts = evaluate_bezier(b1_cp, t)  # (N, K, 2)

    # Scale to pixel space for meaningful curvature magnitudes
    scale = torch.tensor([W, H], device=boundary_cp.device, dtype=boundary_cp.dtype)
    b0_pts = (b0_pts + 1) / 2 * scale
    b1_pts = (b1_pts + 1) / 2 * scale

    # Concatenate: boundary 0 forward, boundary 1 reversed (closed loop)
    loop_pts = torch.cat([b0_pts, b1_pts.flip(dims=[1])], dim=1)  # (N, 2K, 2)

    # Second-order finite differences: prev - 2*curr + next
    prev = loop_pts[:, :-2, :]  # (N, 2K-2, 2)
    curr = loop_pts[:, 1:-1, :]
    nxt = loop_pts[:, 2:, :]
    second_diff = prev - 2 * curr + nxt  # (N, 2K-2, 2)

    curvature_mag = second_diff.pow(2).sum(dim=-1)  # (N, 2K-2)

    # Angle threshold mask at 60 degrees: only penalize where angle < 60
    # Compute angle at each interior point from the two adjacent edges
    edge_a = curr - prev  # (N, 2K-2, 2)
    edge_b = nxt - curr   # (N, 2K-2, 2)
    dot_ab = (edge_a * edge_b).sum(dim=-1)  # (N, 2K-2)
    norm_a = torch.sqrt(edge_a.pow(2).sum(dim=-1) + 1e-12)
    norm_b = torch.sqrt(edge_b.pow(2).sum(dim=-1) + 1e-12)
    cos_angle = dot_ab / (norm_a * norm_b)
    cos_angle = cos_angle.clamp(-1, 1)

    # angle < 60 degrees means cos(angle) > cos(60) = 0.5
    angle_mask = (cos_angle > 0.5).float()

    masked = curvature_mag * angle_mask
    if angle_mask.sum() == 0:
        return torch.tensor(0.0, device=boundary_cp.device)

    return masked.mean()


def boundary_joint_loss(
    control_points: Float[Tensor, "N CP 2"],
    degree: int = 3,
) -> Float[Tensor, ""]:
    """Penalize segment joints that exceed [-1, 1] coordinate bounds.

    Extracts control points at segment boundaries (every ``degree`` indices)
    and penalizes any that fall outside the [-1, 1] range.

    Args:
        control_points: (N, CP, 2) control points in [-1, 1] model space.
        degree: Bezier degree, used to identify joint indices.

    Returns:
        Scalar loss value.
    """
    n, num_cp, _ = control_points.shape
    if n == 0:
        return torch.tensor(0.0, device=control_points.device)

    # Joint indices: every `degree` CPs (segment boundaries)
    joint_indices = list(range(0, num_cp, degree))
    if (num_cp - 1) not in joint_indices:
        joint_indices.append(num_cp - 1)

    joints = control_points[:, joint_indices, :]  # (N, J, 2)
    loss = (F.relu(joints - 1) + F.relu(-1 - joints)).mean()
    return loss


# ── Xing Loss (LIVE method) ─────────────────────────────────────────────


def _sine_theta(a: Float[Tensor, "N 2"], b: Float[Tensor, "N 2"]) -> Float[Tensor, " N"]:
    """Signed sine of the angle between 2D vector pairs.

    Args:
        a, b: (N, 2) vectors.

    Returns:
        sin(theta) for each pair, shape (N,).
    """
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    norms = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
    return cross / norms


def _xing_loss_cubic(
    p0: Float[Tensor, "N 2"],
    p1: Float[Tensor, "N 2"],
    p2: Float[Tensor, "N 2"],
    p3: Float[Tensor, "N 2"],
) -> Float[Tensor, " N"]:
    """LIVE Xing loss for a batch of cubic Bezier segments.

    Penalizes self-intersecting control polygons using direction-gated
    sine penalty: if the middle CP crosses to the wrong side of the
    chord (p0->p3), penalize.

    Args:
        p0, p1, p2, p3: Control points, each (N, 2).

    Returns:
        Per-segment loss (N,).
    """
    cs1 = p1 - p0  # (N, 2)
    cs2 = p2 - p0
    cs3 = p3 - p0

    sina = _sine_theta(cs1, cs3)   # sine of angle between first edge and chord
    sin12 = _sine_theta(cs1, cs2)  # sine of angle between first edge and second edge

    direct = (sin12 >= 0).float()
    opst = 1.0 - direct

    loss = direct * F.relu(-sina) + opst * F.relu(sina)
    return loss


def xing_loss(scene: VectorGraphicsScene) -> Float[Tensor, ""]:
    """Total Xing loss for all curves in the scene.

    Open curves: 3 cubic segments each (CPs [0:4], [3:7], [6:10]).
    Closed curves: 1 cubic per boundary x 2 boundaries (when 4 CPs),
        or sliding window of cubics for higher-order boundaries.
    """
    losses: list[Tensor] = []

    # Open curves: 3 segments per curve
    if scene.n_open > 0:
        cp = scene.open_control_points  # (N, 10, 2)
        for seg_start in [0, 3, 6]:
            loss = _xing_loss_cubic(
                cp[:, seg_start], cp[:, seg_start + 1],
                cp[:, seg_start + 2], cp[:, seg_start + 3],
            )
            losses.append(loss)

    # Closed curves: per-boundary cubics
    if scene.n_closed > 0:
        bcp = scene.closed_boundary_cp  # (N, 2, num_cp, 2)
        num_cp = bcp.shape[2]
        if num_cp == 4:
            for b in range(2):
                loss = _xing_loss_cubic(
                    bcp[:, b, 0], bcp[:, b, 1],
                    bcp[:, b, 2], bcp[:, b, 3],
                )
                losses.append(loss)
        elif num_cp > 4:
            # Sliding window of cubics along each boundary
            for b in range(2):
                for i in range(num_cp - 3):
                    loss = _xing_loss_cubic(
                        bcp[:, b, i], bcp[:, b, i + 1],
                        bcp[:, b, i + 2], bcp[:, b, i + 3],
                    )
                    losses.append(loss)

    if not losses:
        device = next(scene.parameters()).device
        return torch.tensor(0.0, device=device)

    return torch.cat(losses).sum()


# ── Main Entry Point ─────────────────────────────────────────────────────


def compute_loss(
    rendered: Float[Tensor, "C H W"],
    target: Float[Tensor, "C H W"],
    scene: VectorGraphicsScene,
    config: LossConfig,
    step: int = 0,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Compute all enabled loss terms and return the total.

    Args:
        rendered: Rendered image (3, H, W) in [0, 1].
        target: Target image (3, H, W) in [0, 1].
        scene: The VectorGraphicsScene being optimized.
        config: Loss configuration controlling which terms are active.
        step: Current optimization step (unused for now, reserved for scheduling).

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict maps term names
        to scalar float values for logging/debugging.
    """
    device = rendered.device
    loss_dict: dict[str, float] = {}

    # Reconstruction loss (always applied)
    recon = reconstruction_loss(rendered, target, config.loss_type)
    total = recon
    loss_dict["reconstruction"] = recon.item()

    # Xing loss
    if config.lambda_xing > 0:
        xing = xing_loss(scene)
        total = total + config.lambda_xing * xing
        loss_dict["xing"] = xing.item()

    # Shape regularizer (closed curves only)
    if config.apply_shape_reg and config.lambda_shape > 0 and scene.n_closed > 0:
        bcp = scene.closed_boundary_cp  # (N, 2, CP, 2)
        # Apply to each boundary independently
        b0_loss = shape_regularizer(bcp[:, 0, :, :])
        b1_loss = shape_regularizer(bcp[:, 1, :, :])
        shape_loss = (b0_loss + b1_loss) / 2
        total = total + config.lambda_shape * shape_loss
        loss_dict["shape_reg"] = shape_loss.item()

    # Opacity prior (closed curves only)
    if config.apply_opacity_prior and config.lambda_opacity_prior > 0 and scene.n_closed > 0:
        op_loss = opacity_prior(scene.closed_opacities)
        total = total + config.lambda_opacity_prior * op_loss
        loss_dict["opacity_prior"] = op_loss.item()

    # Curvature loss (closed curves only)
    if config.apply_curvature and config.lambda_curvature > 0 and scene.n_closed > 0:
        _, H, W = target.shape
        bcp = scene.closed_boundary_cp
        curv_loss = curvature_loss(bcp, H, W)
        total = total + config.lambda_curvature * curv_loss
        loss_dict["curvature"] = curv_loss.item()

    # Boundary joint loss (both curve types)
    if config.apply_boundary and config.lambda_boundary > 0:
        joint_losses: list[Tensor] = []
        if scene.n_open > 0:
            joint_losses.append(boundary_joint_loss(scene.open_control_points, degree=3))
        if scene.n_closed > 0:
            bcp = scene.closed_boundary_cp
            joint_losses.append(boundary_joint_loss(bcp[:, 0, :, :], degree=3))
            joint_losses.append(boundary_joint_loss(bcp[:, 1, :, :], degree=3))
        if joint_losses:
            bnd_loss = torch.stack(joint_losses).mean()
            total = total + config.lambda_boundary * bnd_loss
            loss_dict["boundary_joint"] = bnd_loss.item()

    loss_dict["total"] = total.item()

    return total, loss_dict
