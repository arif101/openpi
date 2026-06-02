"""Shared transition builder with frame-stacking.

stack=1 : single frame (no velocity in the representation).
stack=2 : two consecutive frames concatenated on channels -> velocity becomes
          observable, matching V-JEPA's use of short clips. This is what makes
          momentum-violation anomalies (freeze) detectable.

A "state" at time t is the stack ending at t: [img_{t-stack+1} .. img_t].
A transition is (state_t, a_t, state_{t+1}) where state_{t+1} ends at t+1.
"""
from __future__ import annotations

import numpy as np
import torch


def stack_state(imgs_seq, t, stack):
    """imgs_seq: (T+1,H,W,3) uint8 ; returns (3*stack,H,W) uint8 channels-first."""
    frames = [imgs_seq[max(0, t - k)] for k in range(stack - 1, -1, -1)]
    s = np.concatenate(frames, axis=-1)  # (H,W,3*stack)
    return np.transpose(s, (2, 0, 1))


def build_stacked_transitions(path, stack=2):
    d = np.load(path)
    imgs, acts, ball = d["images"], d["actions"], d["ball"]
    N, T1 = imgs.shape[0], imgs.shape[1]
    T = T1 - 1
    st, st1, a_list, ball_list = [], [], [], []
    for r in range(N):
        for t in range(T):  # transition t -> t+1
            st.append(stack_state(imgs[r], t, stack))
            st1.append(stack_state(imgs[r], t + 1, stack))
            a_list.append(acts[r, t])
            ball_list.append(ball[r, t])
    st = torch.from_numpy(np.stack(st)).contiguous()
    st1 = torch.from_numpy(np.stack(st1)).contiguous()
    a = torch.from_numpy(np.stack(a_list).astype(np.float32))
    ball = torch.from_numpy(np.stack(ball_list).astype(np.float32))
    return st, st1, a, ball


def to_float(x_uint8, device):
    return x_uint8.to(device).float().div_(255.0)
