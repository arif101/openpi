"""PyTorch autograd wrapper for differentiable MuJoCo dynamics.

Wraps MuJoCo's mj_step + mjd_transitionFD inside a torch.autograd.Function so
gradients flow through physics rollouts. This is the Phase C primitive — it
lets us fine-tune a VLA policy with physics-grounded gradients flowing all
the way from a final-state cost back through the rollout into the policy.

Math:
  x_{t+1} = f(x_t, u_t)               (MuJoCo step)
  forward:  takes (x, u) → x_next; caches Jacobians A=∂x_next/∂x, B=∂x_next/∂u
  backward: grad_x = A^T grad_x_next,  grad_u = B^T grad_x_next

Multi-step rollout chains these in the autograd graph automatically.

Jacobians come from MuJoCo's mjd_transitionFD (centered finite differences).
Cost: ~(n_x + n_u) extra mj_step calls per Jacobian. For LIBERO this is
~129 extra steps × ~1μs = ~130μs per step's Jacobian.

Not used by Phase A (MPPI) or Phase B (LoRA distillation). Phase C only.
Shipped early so the primitive's ready when we get there.
"""

from __future__ import annotations

from typing import Tuple

import mujoco
import numpy as np
import torch


class MuJoCoStepFunction(torch.autograd.Function):
    """One differentiable MuJoCo step.

    forward(x, u, model, data) → x_next
        - x: [n_x] state tensor (qpos concatenated with qvel)
        - u: [n_u] control tensor
        - model, data: MuJoCo model and persistent data buffer
        - Returns: [n_x] state tensor after one mj_step

    The data buffer is MUTATED to the post-step state. Sequential calls
    advance it. To start a fresh rollout, set qpos/qvel of data explicitly
    or use MuJoCoRollout below.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, u: torch.Tensor, model: mujoco.MjModel,
                data: mujoco.MjData) -> torch.Tensor:
        # MuJoCo's "linearized state" for transition Jacobians is 2*nv, not
        # nq+nv. They agree for hinge/slide-only models (where nq == nv) but
        # differ for free/ball joints which embed orientation as quaternion
        # in qpos. We expose the 2*nv state to keep autograd consistent with
        # what mjd_transitionFD differentiates.
        nq = model.nq
        nv = model.nv
        n_x = 2 * nv
        n_u = model.nu

        if x.shape[-1] != n_x:
            raise ValueError(
                f"x dim {x.shape[-1]} != n_x = 2*nv = {n_x}. "
                f"State convention: x = [qpos_velocity_aligned, qvel]. "
                f"For hinge/slide-only models nq==nv so it's just [qpos, qvel]."
            )
        if u.shape[-1] != n_u:
            raise ValueError(f"u dim {u.shape[-1]} != n_u {n_u}")

        dtype = x.dtype
        device = x.device

        # Set pre-step state. For hinge/slide-only models nq == nv so this is
        # straightforward. For free/ball joints, the first nq elements of x
        # are interpreted as qpos; the user is responsible for ensuring it's
        # a valid configuration (e.g., normalized quaternions). Phase A/B
        # don't use this wrapper, so this complication doesn't bite yet.
        x_np = x.detach().cpu().numpy().astype(np.float64)
        u_np = u.detach().cpu().numpy().astype(np.float64)

        if nq == nv:
            data.qpos[:] = x_np[:nq]
        else:
            # Caller must pre-pack a valid qpos (quaternion-aware).
            # We assume the first nq entries of x are valid qpos.
            data.qpos[:] = x_np[:nq]
        data.qvel[:] = x_np[-nv:]
        data.ctrl[:] = u_np

        # Compute Jacobians A = ∂x_next/∂x, B = ∂x_next/∂u via centered FD.
        # mjd_transitionFD does NOT advance the state — it leaves `data` at
        # the pre-step configuration. We explicitly mj_step afterward to get
        # the post-step state.
        A = np.zeros((n_x, n_x), dtype=np.float64)
        B = np.zeros((n_x, n_u), dtype=np.float64)
        mujoco.mjd_transitionFD(model, data, 1e-6, 1, A, B, None, None)

        # Now actually advance one step to get the next state
        mujoco.mj_step(model, data)
        if nq == nv:
            x_next_np = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        else:
            x_next_np = np.concatenate([data.qpos.copy(), data.qvel.copy()])

        ctx.save_for_backward(
            torch.as_tensor(A, dtype=dtype, device=device),
            torch.as_tensor(B, dtype=dtype, device=device),
        )
        return torch.as_tensor(x_next_np, dtype=dtype, device=device)

    @staticmethod
    def backward(ctx, grad_x_next: torch.Tensor) -> Tuple:
        A, B = ctx.saved_tensors
        grad_x = A.T @ grad_x_next
        grad_u = B.T @ grad_x_next
        # model, data are non-differentiable
        return grad_x, grad_u, None, None


class MuJoCoRollout:
    """Multi-step differentiable rollout wrapper.

    Usage:
        rollout = MuJoCoRollout(mj_model)
        x_init = torch.tensor(np.concatenate([qpos0, qvel0]), dtype=torch.float64)
        u = torch.zeros((H, mj_model.nu), dtype=torch.float64, requires_grad=True)

        states = rollout(x_init, u)         # [H+1, n_x]
        loss = torch.linalg.norm(states[-1] - target_state)
        loss.backward()                     # u.grad now populated

        # Gradient descent on u:
        with torch.no_grad():
            u -= 0.01 * u.grad
            u.grad.zero_()

    The internal MjData is reset to ``x_init`` at the start of each ``__call__``,
    so successive rollouts don't leak state. Note: not thread-safe (each thread
    needs its own MuJoCoRollout instance).
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.nq = model.nq
        self.nv = model.nv
        self.n_x = 2 * self.nv  # 2*nv is the linearized state dim for transitionFD
        self.n_u = model.nu

    def __call__(self, x_init: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Roll out action_chunk from x_init, return full state trajectory.

        Args:
            x_init: [n_x] initial state (qpos concatenated with qvel)
            action_chunk: [H, n_u] sequence of controls

        Returns:
            [H+1, n_x] state trajectory including the initial state at index 0.
        """
        if action_chunk.dim() != 2:
            raise ValueError(f"action_chunk must be 2-D [H, n_u], got {action_chunk.shape}")
        H = action_chunk.shape[0]
        x = x_init
        states = [x]
        for t in range(H):
            x = MuJoCoStepFunction.apply(x, action_chunk[t], self.model, self.data)
            states.append(x)
        return torch.stack(states, dim=0)
