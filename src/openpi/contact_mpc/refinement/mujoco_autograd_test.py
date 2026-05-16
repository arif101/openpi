"""Tests for the MuJoCo PyTorch autograd wrapper."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from openpi.contact_mpc.refinement.mujoco_autograd import (
    MuJoCoRollout,
    MuJoCoStepFunction,
)


# Same 2-DOF arm + free-body box scene used elsewhere
TEST_XML = """
<mujoco model="test_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-1.57 1.57"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" limited="true" range="-1.57 1.57"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
        <body name="eef" pos="0.3 0 0">
          <geom name="grip" type="box" size="0.03 0.03 0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-1 1" gear="5"/>
    <motor joint="j2" ctrlrange="-1 1" gear="5"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(TEST_XML)


def test_forward_output_shape(model):
    rollout = MuJoCoRollout(model)
    x_init = torch.zeros(rollout.n_x, dtype=torch.float64)
    u = torch.zeros(1, rollout.n_u, dtype=torch.float64)
    states = rollout(x_init, u)
    assert states.shape == (2, rollout.n_x)
    assert torch.all(torch.isfinite(states))


def test_forward_zero_action_state_changes_slowly(model):
    """Under gravity, zero action lets the arm fall slowly. State should change
    in a finite + finite-magnitude way."""
    rollout = MuJoCoRollout(model)
    x_init = torch.zeros(rollout.n_x, dtype=torch.float64)
    u = torch.zeros(5, rollout.n_u, dtype=torch.float64)
    states = rollout(x_init, u)
    delta = (states[-1] - states[0]).abs().max().item()
    assert 0.0 <= delta < 1.0  # bounded change


def test_backward_returns_gradients(model):
    rollout = MuJoCoRollout(model)
    x_init = torch.zeros(rollout.n_x, dtype=torch.float64)
    u = torch.full((3, rollout.n_u), 0.1, dtype=torch.float64, requires_grad=True)
    states = rollout(x_init, u)
    loss = torch.linalg.norm(states[-1])
    loss.backward()
    assert u.grad is not None
    assert u.grad.shape == u.shape
    assert torch.all(torch.isfinite(u.grad))
    # Gradient should be NONZERO (the action affects the final state)
    assert u.grad.abs().sum() > 1e-6


def test_gradient_matches_finite_difference(model):
    """Sanity-check: autograd gradient ≈ central-difference numerical gradient.

    We perturb each control entry by eps and compare the finite-difference
    derivative with what autograd returns. Both come from finite differences
    underneath (since mjd_transitionFD uses FD), but at different epsilons —
    so they should agree closely.
    """
    rollout = MuJoCoRollout(model)
    torch.manual_seed(0)
    x_init = torch.zeros(rollout.n_x, dtype=torch.float64)
    H = 3
    u = torch.full((H, rollout.n_u), 0.05, dtype=torch.float64, requires_grad=True)

    # Define a scalar loss
    def loss_fn(u_val):
        traj = rollout(x_init, u_val)
        return torch.linalg.norm(traj[-1])

    loss = loss_fn(u)
    loss.backward()
    grad_autograd = u.grad.detach().clone()

    # Numerical gradient via central differences (separate larger eps)
    eps = 1e-4
    grad_numerical = torch.zeros_like(u)
    with torch.no_grad():
        for t in range(H):
            for j in range(rollout.n_u):
                u_pos = u.detach().clone()
                u_pos[t, j] += eps
                loss_pos = loss_fn(u_pos).item()
                u_neg = u.detach().clone()
                u_neg[t, j] -= eps
                loss_neg = loss_fn(u_neg).item()
                grad_numerical[t, j] = (loss_pos - loss_neg) / (2 * eps)

    # Allow some tolerance — both methods use finite differences internally
    # and the dynamics are nonlinear; expect agreement to ~1e-3 relative.
    rel_error = (grad_autograd - grad_numerical).abs().max().item() / (
        grad_numerical.abs().max().item() + 1e-9
    )
    assert rel_error < 1e-2, (
        f"autograd vs numerical relative error {rel_error:.4e} too large"
    )


def test_gradient_descent_actually_reduces_loss(model):
    """End-to-end smoke: gradient steps on u should reduce a target-distance loss.

    This is the property we actually care about for Phase C — that physics
    gradients let us optimize actions to satisfy a target.

    With the synthetic 2-DOF arm at 2ms timestep and gear=5, the dynamics
    move slowly. We pick a small target reachable in the horizon and a
    large learning rate so the test is decisive in few iterations.
    """
    rollout = MuJoCoRollout(model)
    x_init = torch.zeros(rollout.n_x, dtype=torch.float64)
    H = 50  # 50 × 2ms = 100ms simulation window
    u = torch.zeros((H, rollout.n_u), dtype=torch.float64, requires_grad=True)

    # Small target reachable within the horizon under bounded actions
    target_qpos_j1 = 0.05
    losses = []
    optimizer = torch.optim.SGD([u], lr=200.0)  # large lr — small gradients
    for step in range(20):
        optimizer.zero_grad()
        traj = rollout(x_init, u)
        final_qpos_j1 = traj[-1, 0]  # qpos[0] = j1 angle
        loss = (final_qpos_j1 - target_qpos_j1) ** 2
        loss.backward()
        optimizer.step()
        # Clip actions to valid range
        with torch.no_grad():
            u.clamp_(-1.0, 1.0)
        losses.append(loss.item())

    # Loss should monotonically decrease (modulo numerical noise)
    assert losses[-1] < losses[0], (
        f"loss not decreasing across 20 steps: first={losses[0]}, last={losses[-1]}"
    )
    # And we should reduce loss by at least half
    assert losses[-1] < losses[0] / 2, f"loss did not halve over 20 iters: {losses[::4]}"


def test_shape_mismatch_raises(model):
    rollout = MuJoCoRollout(model)
    # Wrong x dim
    bad_x = torch.zeros(rollout.n_x + 1, dtype=torch.float64)
    u = torch.zeros((1, rollout.n_u), dtype=torch.float64)
    with pytest.raises(ValueError, match="n_x"):
        rollout(bad_x, u)

    # Wrong u dim
    x = torch.zeros(rollout.n_x, dtype=torch.float64)
    bad_u = torch.zeros((1, rollout.n_u + 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="n_u"):
        rollout(x, bad_u)
