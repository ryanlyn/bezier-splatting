"""Adan optimizer — Adaptive Nesterov Momentum Algorithm.

Implements the Adan optimizer from "Adan: Adaptive Nesterov Momentum Algorithm
for Faster Optimizing Deep Models" (Xie et al., 2023). Uses three momentum
terms and a gradient-difference second moment for adaptive learning.

Default hyperparameters match the original Bezier Splatting paper.
"""

import torch
from torch.optim.optimizer import Optimizer


class Adan(Optimizer):
    r"""Adan optimizer with Nesterov-style momentum.

    Update rule per parameter at step t (with bias corrections
    bc_i = 1 - (1 - b_i)^t, matching the official implementation)::

        m_t = (1 - b1) * m_{t-1} + b1 * g_t
        v_t = (1 - b2) * v_{t-1} + b2 * (g_t - g_{t-1})
        n_t = (1 - b3) * n_{t-1} + b3 * (g_t + (1 - b2) * (g_t - g_{t-1}))^2
        update = (m_t / bc1 + (1 - b2) * v_t / bc2) / (sqrt(n_t / bc3) + eps)
        theta_t = (theta_{t-1} - lr * update) / (1 + lr * wd)

    Weight decay uses the proximal form from the paper.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients (b1, b2, b3) for moment estimation (default: (0.98, 0.92, 0.99)).
        eps: Numerical stability term (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta3: {betas[2]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                the loss. Not used in typical Bezier Splatting training.

        Returns:
            Loss value if closure was provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients")

                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Store negative of previous gradient for computing g_t - g_{t-1}
                    state["neg_pre_grad"] = grad.neg().clone()

                state["step"] += 1
                t = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_diff = state["exp_avg_diff"]
                exp_avg_sq = state["exp_avg_sq"]
                neg_pre_grad = state["neg_pre_grad"]

                # Bias corrections (official Adan): bc_i = 1 - (1 - b_i)^t
                bias_correction1 = 1.0 - (1.0 - beta1) ** t
                bias_correction2 = 1.0 - (1.0 - beta2) ** t
                bias_correction3 = 1.0 - (1.0 - beta3) ** t

                # g_t - g_{t-1} (neg_pre_grad stores -g_{t-1})
                diff = grad + neg_pre_grad

                # m_t = (1 - b1) * m_{t-1} + b1 * g_t
                exp_avg.lerp_(grad, beta1)

                # v_t = (1 - b2) * v_{t-1} + b2 * (g_t - g_{t-1})
                exp_avg_diff.lerp_(diff, beta2)

                # n_t = (1 - b3) * n_{t-1} + b3 * (g_t + (1 - b2) * (g_t - g_{t-1}))^2
                update_sq = (grad + (1.0 - beta2) * diff).square()
                exp_avg_sq.lerp_(update_sq, beta3)

                # update = (m̂_t + (1 - b2) * v̂_t) / (sqrt(n̂_t) + eps)
                denom = (exp_avg_sq / bias_correction3).sqrt().add_(eps)
                update = (
                    exp_avg / bias_correction1
                    + (1.0 - beta2) * exp_avg_diff / bias_correction2
                ).div_(denom)
                p.add_(update, alpha=-lr)

                # Proximal weight decay: theta_t = theta_t' / (1 + lr * wd)
                if weight_decay != 0.0:
                    p.div_(1.0 + lr * weight_decay)

                # Store -g_t for next step
                neg_pre_grad.copy_(grad.neg())

        return loss
