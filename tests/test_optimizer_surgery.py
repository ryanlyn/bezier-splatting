"""Tests for optimizer state surgery (prune/extend/splice).

Verifies that momentum tensors are correctly sliced, extended, and spliced
when curves are pruned or densified, preserving optimizer state for surviving
curves instead of resetting to zero.
"""

import torch
import torch.nn as nn

from bezier_splatting.adan import Adan
from bezier_splatting.optimization import (
    _extend_optimizer_state,
    _prune_optimizer_state,
    _splice_optimizer_state,
)


def _make_optimizer_with_state(
    param: nn.Parameter,
    optimizer_type: str = "adan",
    lr: float = 0.01,
    n_steps: int = 3,
) -> torch.optim.Optimizer:
    """Create an optimizer, run a few steps to populate state, return it."""
    if optimizer_type == "adan":
        opt = Adan([{"params": [param], "lr": lr}], betas=(0.98, 0.92, 0.99))
    else:
        opt = torch.optim.Adam([{"params": [param], "lr": lr}])

    for _ in range(n_steps):
        opt.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        opt.step()

    return opt


# ── Prune tests ──────────────────────────────────────────────────────────


class TestPruneOptimizerState:
    """Tests for _prune_optimizer_state."""

    def test_shapes_match_after_prune_adan(self):
        """After pruning, state tensor shapes match the new parameter shape."""
        param = nn.Parameter(torch.randn(10, 5))
        opt = _make_optimizer_with_state(param, "adan")

        mask = torch.tensor([True, False, True, True, False, False, True, True, False, True])
        new_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, new_param, mask)

        state = opt.state[new_param]
        assert state["exp_avg"].shape == (6, 5)
        assert state["exp_avg_sq"].shape == (6, 5)
        assert state["exp_avg_diff"].shape == (6, 5)
        assert state["neg_pre_grad"].shape == (6, 5)

    def test_shapes_match_after_prune_adam(self):
        """After pruning, state tensor shapes match for Adam optimizer."""
        param = nn.Parameter(torch.randn(8, 3))
        opt = _make_optimizer_with_state(param, "adam")

        mask = torch.tensor([True, True, False, True, False, False, True, True])
        new_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, new_param, mask)

        state = opt.state[new_param]
        assert state["exp_avg"].shape == (5, 3)
        assert state["exp_avg_sq"].shape == (5, 3)

    def test_values_preserved_for_survivors(self):
        """State values for kept curves are identical to the original."""
        param = nn.Parameter(torch.randn(5, 3))
        opt = _make_optimizer_with_state(param, "adan")

        # Save original state values for comparison
        orig_exp_avg = opt.state[param]["exp_avg"].clone()
        orig_exp_avg_sq = opt.state[param]["exp_avg_sq"].clone()
        orig_step = opt.state[param]["step"]

        mask = torch.tensor([True, False, True, False, True])
        new_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, new_param, mask)

        state = opt.state[new_param]
        assert torch.allclose(state["exp_avg"], orig_exp_avg[mask])
        assert torch.allclose(state["exp_avg_sq"], orig_exp_avg_sq[mask])
        assert state["step"] == orig_step

    def test_old_param_removed_from_state(self):
        """After prune, the old parameter key is removed from optimizer.state."""
        param = nn.Parameter(torch.randn(4, 2))
        opt = _make_optimizer_with_state(param, "adan")

        mask = torch.tensor([True, True, False, True])
        new_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, new_param, mask)

        assert param not in opt.state
        assert new_param in opt.state

    def test_param_group_updated(self):
        """After prune, param_groups references point to the new parameter."""
        param = nn.Parameter(torch.randn(6, 2))
        opt = _make_optimizer_with_state(param, "adan")

        mask = torch.tensor([True, False, True, True, False, True])
        new_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, new_param, mask)

        found = False
        for group in opt.param_groups:
            for p in group["params"]:
                if p is new_param:
                    found = True
                assert p is not param
        assert found

    def test_no_state_noop(self):
        """Prune updates param refs even when optimizer state has not been created."""
        param = nn.Parameter(torch.randn(4, 2))
        opt = Adan([{"params": [param], "lr": 0.01}])
        # No steps taken, so no state

        mask = torch.tensor([True, False, True, True])
        new_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, new_param, mask)

        # No state is created, but param refs must still be updated.
        assert new_param not in opt.state
        assert opt.param_groups[0]["params"][0] is new_param
        assert opt.param_groups[0]["params"][0] is not param

    def test_step_counter_preserved(self):
        """Step counter is preserved across prune, not reset to 0."""
        param = nn.Parameter(torch.randn(4, 2))
        opt = _make_optimizer_with_state(param, "adan", n_steps=7)

        mask = torch.tensor([True, True, False, True])
        new_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, new_param, mask)

        assert opt.state[new_param]["step"] == 7

    def test_multidim_param(self):
        """Prune works for higher-dimensional params like (N, 10, 2)."""
        param = nn.Parameter(torch.randn(5, 10, 2))
        opt = _make_optimizer_with_state(param, "adan")

        mask = torch.tensor([True, False, True, True, False])
        new_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, new_param, mask)

        state = opt.state[new_param]
        assert state["exp_avg"].shape == (3, 10, 2)
        assert state["exp_avg_sq"].shape == (3, 10, 2)


# ── Extend tests ─────────────────────────────────────────────────────────


class TestExtendOptimizerState:
    """Tests for _extend_optimizer_state."""

    def test_shapes_after_extend_adan(self):
        """After extend, state tensors grow by n_new along dim 0."""
        param = nn.Parameter(torch.randn(5, 3))
        opt = _make_optimizer_with_state(param, "adan")

        n_new = 3
        extension = torch.randn(n_new, 3)
        new_param = nn.Parameter(torch.cat([param, extension]))
        _extend_optimizer_state(opt, param, new_param, n_new)

        state = opt.state[new_param]
        assert state["exp_avg"].shape == (8, 3)
        assert state["exp_avg_sq"].shape == (8, 3)
        assert state["exp_avg_diff"].shape == (8, 3)
        assert state["neg_pre_grad"].shape == (8, 3)

    def test_shapes_after_extend_adam(self):
        """After extend, state tensors grow correctly for Adam."""
        param = nn.Parameter(torch.randn(4, 2))
        opt = _make_optimizer_with_state(param, "adam")

        n_new = 2
        extension = torch.randn(n_new, 2)
        new_param = nn.Parameter(torch.cat([param, extension]))
        _extend_optimizer_state(opt, param, new_param, n_new)

        state = opt.state[new_param]
        assert state["exp_avg"].shape == (6, 2)
        assert state["exp_avg_sq"].shape == (6, 2)

    def test_existing_values_preserved(self):
        """Original state values are preserved in the first N entries."""
        param = nn.Parameter(torch.randn(4, 3))
        opt = _make_optimizer_with_state(param, "adan")

        orig_exp_avg = opt.state[param]["exp_avg"].clone()
        orig_step = opt.state[param]["step"]

        n_new = 2
        extension = torch.randn(n_new, 3)
        new_param = nn.Parameter(torch.cat([param, extension]))
        _extend_optimizer_state(opt, param, new_param, n_new)

        state = opt.state[new_param]
        assert torch.allclose(state["exp_avg"][:4], orig_exp_avg)
        assert state["step"] == orig_step

    def test_new_entries_are_zero(self):
        """Extended entries (new curves) have zero-initialized state."""
        param = nn.Parameter(torch.randn(3, 2))
        opt = _make_optimizer_with_state(param, "adan")

        n_new = 4
        extension = torch.randn(n_new, 2)
        new_param = nn.Parameter(torch.cat([param, extension]))
        _extend_optimizer_state(opt, param, new_param, n_new)

        state = opt.state[new_param]
        for key in ("exp_avg", "exp_avg_sq", "exp_avg_diff", "neg_pre_grad"):
            assert (state[key][3:] == 0).all(), f"{key} new entries should be zero"

    def test_old_param_removed(self):
        """Old parameter is removed from state and param_groups."""
        param = nn.Parameter(torch.randn(3, 2))
        opt = _make_optimizer_with_state(param, "adan")

        n_new = 2
        new_param = nn.Parameter(torch.cat([param, torch.randn(n_new, 2)]))
        _extend_optimizer_state(opt, param, new_param, n_new)

        assert param not in opt.state
        assert new_param in opt.state

    def test_no_state_noop(self):
        """Extend updates param refs even when optimizer state has not been created."""
        param = nn.Parameter(torch.randn(3, 2))
        opt = Adan([{"params": [param], "lr": 0.01}])

        n_new = 2
        new_param = nn.Parameter(torch.cat([param, torch.randn(n_new, 2)]))
        _extend_optimizer_state(opt, param, new_param, n_new)

        assert new_param not in opt.state
        assert opt.param_groups[0]["params"][0] is new_param
        assert opt.param_groups[0]["params"][0] is not param


# ── Splice tests ─────────────────────────────────────────────────────────


class TestSpliceOptimizerState:
    """Tests for _splice_optimizer_state (depth parameter)."""

    def test_shapes_after_splice(self):
        """After splice, state tensors have n_new entries inserted at insert_idx."""
        param = nn.Parameter(torch.randn(6, 1))
        opt = _make_optimizer_with_state(param, "adan")

        n_new = 3
        insert_idx = 2
        new_data = torch.cat([param[:insert_idx], torch.ones(n_new, 1), param[insert_idx:]])
        new_param = nn.Parameter(new_data)
        _splice_optimizer_state(opt, param, new_param, insert_idx, n_new)

        state = opt.state[new_param]
        assert state["exp_avg"].shape == (9, 1)
        assert state["exp_avg_sq"].shape == (9, 1)

    def test_values_preserved_around_splice(self):
        """Original values are preserved before and after the splice point."""
        param = nn.Parameter(torch.randn(6, 1))
        opt = _make_optimizer_with_state(param, "adan")

        orig_exp_avg = opt.state[param]["exp_avg"].clone()
        insert_idx = 2
        n_new = 3

        new_data = torch.cat([param[:insert_idx], torch.ones(n_new, 1), param[insert_idx:]])
        new_param = nn.Parameter(new_data)
        _splice_optimizer_state(opt, param, new_param, insert_idx, n_new)

        state = opt.state[new_param]
        # Before splice point
        assert torch.allclose(state["exp_avg"][:insert_idx], orig_exp_avg[:insert_idx])
        # After splice point
        assert torch.allclose(state["exp_avg"][insert_idx + n_new:], orig_exp_avg[insert_idx:])

    def test_spliced_entries_are_zero(self):
        """Inserted entries are zero-initialized."""
        param = nn.Parameter(torch.randn(4, 1))
        opt = _make_optimizer_with_state(param, "adan")

        insert_idx = 2
        n_new = 3
        new_data = torch.cat([param[:insert_idx], torch.ones(n_new, 1), param[insert_idx:]])
        new_param = nn.Parameter(new_data)
        _splice_optimizer_state(opt, param, new_param, insert_idx, n_new)

        state = opt.state[new_param]
        for key in ("exp_avg", "exp_avg_sq", "exp_avg_diff", "neg_pre_grad"):
            spliced = state[key][insert_idx:insert_idx + n_new]
            assert (spliced == 0).all(), f"{key} spliced entries should be zero"

    def test_no_state_still_references_new_param(self):
        """Splice updates param refs even when optimizer state has not been created."""
        param = nn.Parameter(torch.randn(4, 1))
        opt = Adan([{"params": [param], "lr": 0.01}])

        insert_idx = 2
        n_new = 2
        new_param = nn.Parameter(torch.cat([param[:insert_idx], torch.ones(n_new, 1), param[insert_idx:]]))
        _splice_optimizer_state(opt, param, new_param, insert_idx, n_new)

        assert new_param not in opt.state
        assert opt.param_groups[0]["params"][0] is new_param
        assert opt.param_groups[0]["params"][0] is not param


# ── Round-trip tests ─────────────────────────────────────────────────────


class TestPruneThenExtend:
    """Full prune+extend round-trip preserves surviving state."""

    def test_prune_then_extend_roundtrip_adan(self):
        """Prune 2 of 5, then extend by 2. Surviving state is preserved."""
        param = nn.Parameter(torch.randn(5, 3))
        opt = _make_optimizer_with_state(param, "adan", n_steps=5)

        orig_exp_avg = opt.state[param]["exp_avg"].clone()
        orig_step = opt.state[param]["step"]

        # Prune: keep indices [0, 2, 4]
        mask = torch.tensor([True, False, True, False, True])
        pruned_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, pruned_param, mask)

        # Extend by 2
        n_new = 2
        extended_param = nn.Parameter(torch.cat([pruned_param, torch.randn(n_new, 3)]))
        _extend_optimizer_state(opt, pruned_param, extended_param, n_new)

        state = opt.state[extended_param]
        # Original survivors (now indices 0, 1, 2) match original indices 0, 2, 4
        assert torch.allclose(state["exp_avg"][:3], orig_exp_avg[mask])
        # New entries are zero
        assert (state["exp_avg"][3:] == 0).all()
        # Step preserved
        assert state["step"] == orig_step

    def test_prune_then_extend_roundtrip_adam(self):
        """Same round-trip with Adam optimizer."""
        param = nn.Parameter(torch.randn(6, 2))
        opt = _make_optimizer_with_state(param, "adam", n_steps=5)

        orig_exp_avg = opt.state[param]["exp_avg"].clone()

        mask = torch.tensor([False, True, True, False, True, True])
        pruned_param = nn.Parameter(param[mask].clone())
        _prune_optimizer_state(opt, param, pruned_param, mask)

        n_new = 3
        extended_param = nn.Parameter(torch.cat([pruned_param, torch.randn(n_new, 2)]))
        _extend_optimizer_state(opt, pruned_param, extended_param, n_new)

        state = opt.state[extended_param]
        assert torch.allclose(state["exp_avg"][:4], orig_exp_avg[mask])
        assert (state["exp_avg"][4:] == 0).all()

    def test_multi_group_surgery(self):
        """State surgery works across multiple param groups."""
        p1 = nn.Parameter(torch.randn(4, 3))
        p2 = nn.Parameter(torch.randn(4, 2))
        opt = Adan([
            {"params": [p1], "lr": 0.01, "name": "group1"},
            {"params": [p2], "lr": 0.05, "name": "group2"},
        ])
        for _ in range(3):
            opt.zero_grad()
            loss = (p1**2).sum() + (p2**2).sum()
            loss.backward()
            opt.step()

        orig_p1_avg = opt.state[p1]["exp_avg"].clone()
        orig_p2_avg = opt.state[p2]["exp_avg"].clone()

        # Prune both
        mask = torch.tensor([True, False, True, True])
        new_p1 = nn.Parameter(p1[mask].clone())
        _prune_optimizer_state(opt, p1, new_p1, mask)

        mask2 = torch.tensor([False, True, True, False])
        new_p2 = nn.Parameter(p2[mask2].clone())
        _prune_optimizer_state(opt, p2, new_p2, mask2)

        assert torch.allclose(opt.state[new_p1]["exp_avg"], orig_p1_avg[mask])
        assert torch.allclose(opt.state[new_p2]["exp_avg"], orig_p2_avg[mask2])

        # Verify param groups point to new params (use `is` to avoid tensor eq)
        group_params = [p for g in opt.param_groups for p in g["params"]]
        assert any(p is new_p1 for p in group_params)
        assert any(p is new_p2 for p in group_params)
        assert not any(p is p1 for p in group_params)
        assert not any(p is p2 for p in group_params)


# ── Integration: optimizer continues training after surgery ──────────────


class TestContinuedTraining:
    """Verify optimizer can continue training after state surgery."""

    def test_training_continues_after_prune(self):
        """Optimizer step succeeds after prune surgery."""
        param = nn.Parameter(torch.randn(6, 2))
        opt = _make_optimizer_with_state(param, "adan", n_steps=3)

        mask = torch.tensor([True, False, True, True, False, True])
        new_param = nn.Parameter(param[mask].clone(), requires_grad=True)
        _prune_optimizer_state(opt, param, new_param, mask)

        # Continue training with new param
        for _ in range(5):
            opt.zero_grad()
            loss = (new_param**2).sum()
            loss.backward()
            opt.step()

        # Should have advanced step counter
        assert opt.state[new_param]["step"] == 8  # 3 + 5

    def test_training_continues_after_extend(self):
        """Optimizer step succeeds after extend surgery."""
        param = nn.Parameter(torch.randn(3, 2))
        opt = _make_optimizer_with_state(param, "adan", n_steps=3)

        n_new = 2
        new_param = nn.Parameter(
            torch.cat([param.detach(), torch.randn(n_new, 2)]),
            requires_grad=True,
        )
        _extend_optimizer_state(opt, param, new_param, n_new)

        for _ in range(5):
            opt.zero_grad()
            loss = (new_param**2).sum()
            loss.backward()
            opt.step()

        assert opt.state[new_param]["step"] == 8

    def test_training_continues_after_prune_and_extend(self):
        """Full prune+extend cycle followed by continued training."""
        param = nn.Parameter(torch.randn(8, 3))
        opt = _make_optimizer_with_state(param, "adan", n_steps=5)

        # Prune
        mask = torch.tensor([True, True, False, True, False, True, True, False])
        pruned = nn.Parameter(param[mask].clone(), requires_grad=True)
        _prune_optimizer_state(opt, param, pruned, mask)

        # Extend
        n_new = 3
        extended = nn.Parameter(
            torch.cat([pruned.detach(), torch.randn(n_new, 3)]),
            requires_grad=True,
        )
        _extend_optimizer_state(opt, pruned, extended, n_new)

        # Train more
        for _ in range(10):
            opt.zero_grad()
            loss = (extended**2).sum()
            loss.backward()
            opt.step()

        assert opt.state[extended]["step"] == 15
        # All state tensors match param shape
        for key in ("exp_avg", "exp_avg_sq", "exp_avg_diff", "neg_pre_grad"):
            assert opt.state[extended][key].shape == extended.shape
