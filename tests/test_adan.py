"""Tests for the Adan optimizer."""

import copy

import pytest
import torch

from bezier_splatting.adan import Adan


# ── Basic convergence ──────────────────────────────────────────────────


def test_quadratic_convergence():
    """Adan converges on f(x) = (x - 3)^2 starting from x=0."""
    x = torch.tensor([0.0], requires_grad=True)
    opt = Adan([x], lr=0.05)

    for _ in range(500):
        opt.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        opt.step()

    assert abs(x.item() - 3.0) < 0.05, f"x={x.item()}, expected ~3.0"


def test_multidim_convergence():
    """Adan converges on a 2D quadratic: f(x) = ||x - target||^2."""
    target = torch.tensor([2.0, -1.0])
    x = torch.tensor([0.0, 0.0], requires_grad=True)
    opt = Adan([x], lr=0.05)

    for _ in range(500):
        opt.zero_grad()
        loss = ((x - target) ** 2).sum()
        loss.backward()
        opt.step()

    assert torch.allclose(x, target, atol=0.05), f"x={x.tolist()}, expected {target.tolist()}"


# ── State tensors ──────────────────────────────────────────────────────


def test_state_tensors_exist():
    """After one step, all 4 state tensors exist with correct shapes."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    opt = Adan([x], lr=0.01)

    loss = (x**2).sum()
    loss.backward()
    opt.step()

    state = opt.state[x]
    assert "step" in state and state["step"] == 1
    assert state["exp_avg"].shape == x.shape
    assert state["exp_avg_diff"].shape == x.shape
    assert state["exp_avg_sq"].shape == x.shape
    assert state["neg_pre_grad"].shape == x.shape


def test_state_tensors_nonzero_after_step():
    """State tensors should be nonzero after a step with nonzero gradient."""
    x = torch.tensor([5.0], requires_grad=True)
    opt = Adan([x], lr=0.01)

    loss = x**2
    loss.backward()
    opt.step()

    state = opt.state[x]
    assert state["exp_avg"].abs().sum() > 0
    assert state["exp_avg_sq"].abs().sum() > 0
    # neg_pre_grad stores -g for next iteration
    assert state["neg_pre_grad"].abs().sum() > 0


# ── Weight decay ───────────────────────────────────────────────────────


def test_weight_decay_shrinks_params():
    """Weight decay should push parameters toward zero."""
    x_wd = torch.tensor([10.0], requires_grad=True)
    x_no = torch.tensor([10.0], requires_grad=True)

    opt_wd = Adan([x_wd], lr=0.01, weight_decay=0.1)
    opt_no = Adan([x_no], lr=0.01, weight_decay=0.0)

    # Use zero gradient to isolate weight decay effect
    for _ in range(50):
        opt_wd.zero_grad()
        opt_no.zero_grad()
        # Constant loss => zero gradient
        loss_wd = torch.tensor(0.0, requires_grad=True)
        loss_no = torch.tensor(0.0, requires_grad=True)
        loss_wd.backward()
        loss_no.backward()
        # Manually set grads to zero (loss doesn't depend on x)
        x_wd.grad = torch.zeros_like(x_wd)
        x_no.grad = torch.zeros_like(x_no)
        opt_wd.step()
        opt_no.step()

    # With weight decay, x should be closer to zero
    assert abs(x_wd.item()) < abs(x_no.item()), (
        f"Weight decay param {x_wd.item()} should be smaller than {x_no.item()}"
    )


# ── State dict save/load ──────────────────────────────────────────────


def test_state_dict_roundtrip():
    """Save and load state_dict; training continues correctly."""
    x = torch.tensor([0.0], requires_grad=True)
    opt = Adan([x], lr=0.05)

    # Run a few steps
    for _ in range(100):
        opt.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        opt.step()

    mid_val = x.item()
    saved_state = copy.deepcopy(opt.state_dict())

    # Create fresh optimizer with same param, load state
    x2 = torch.tensor([mid_val], requires_grad=True)
    opt2 = Adan([x2], lr=0.05)
    opt2.load_state_dict(saved_state)

    # Continue training — should converge similarly
    for _ in range(400):
        opt2.zero_grad()
        loss = (x2 - 3.0) ** 2
        loss.backward()
        opt2.step()

    assert abs(x2.item() - 3.0) < 0.05, f"x2={x2.item()} after reload, expected ~3.0"


# ── Comparison with Adam ──────────────────────────────────────────────


def test_both_optimizers_converge():
    """Both Adan and Adam converge on a simple quadratic."""
    target = 5.0

    x_adan = torch.tensor([0.0], requires_grad=True)
    x_adam = torch.tensor([0.0], requires_grad=True)
    opt_adan = Adan([x_adan], lr=0.05)
    opt_adam = torch.optim.Adam([x_adam], lr=0.05)

    for _ in range(500):
        opt_adan.zero_grad()
        loss = (x_adan - target) ** 2
        loss.backward()
        opt_adan.step()

        opt_adam.zero_grad()
        loss = (x_adam - target) ** 2
        loss.backward()
        opt_adam.step()

    assert abs(x_adan.item() - target) < 0.1, f"Adan: x={x_adan.item()}"
    assert abs(x_adam.item() - target) < 0.1, f"Adam: x={x_adam.item()}"


# ── zero_grad ──────────────────────────────────────────────────────────


def test_zero_grad():
    """zero_grad clears gradients (set_to_none=False mode)."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    opt = Adan([x], lr=0.01)

    loss = (x**2).sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0

    opt.zero_grad(set_to_none=False)
    assert x.grad is not None and x.grad.abs().sum() == 0


def test_zero_grad_set_to_none():
    """zero_grad with set_to_none=True (default) sets grad to None."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    opt = Adan([x], lr=0.01)

    loss = (x**2).sum()
    loss.backward()
    assert x.grad is not None

    opt.zero_grad(set_to_none=True)
    assert x.grad is None


# ── Parameter groups ──────────────────────────────────────────────────


def test_per_group_lr():
    """Different param groups can have different learning rates."""
    x = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([0.0], requires_grad=True)

    opt = Adan([
        {"params": [x], "lr": 0.1},
        {"params": [y], "lr": 0.001},
    ])

    for _ in range(100):
        opt.zero_grad()
        loss = (x - 1.0) ** 2 + (y - 1.0) ** 2
        loss.backward()
        opt.step()

    # x with higher LR should be closer to target
    assert abs(x.item() - 1.0) < abs(y.item() - 1.0), (
        f"x={x.item()} should be closer to 1.0 than y={y.item()}"
    )


# ── Validation ─────────────────────────────────────────────────────────


def test_invalid_lr_raises():
    x = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="Invalid learning rate"):
        Adan([x], lr=-0.1)


def test_invalid_betas_raises():
    x = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="Invalid beta1"):
        Adan([x], betas=(1.5, 0.92, 0.99))


def test_closure():
    """Closure-based step returns loss value."""
    x = torch.tensor([5.0], requires_grad=True)
    opt = Adan([x], lr=0.01)

    def closure():
        opt.zero_grad()
        loss = (x - 1.0) ** 2
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert loss.item() > 0
