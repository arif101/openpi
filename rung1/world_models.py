"""Two world models for the structure-vs-coverage test.

MonolithicWM: a flat MLP over the whole (padded) scene vector. No parameter sharing across
  objects, no count-generalization — it memorizes joint configurations (the "coverage" model).

InteractionWM: an object-centric interaction network (Battaglia-style). A SHARED per-object
  dynamics function + pairwise interaction messages, operating on RELATIVE positions. The same
  local rule applies to every object → permutation-equivariant, generalizes to unseen object
  COUNTS and SIZES by construction (the "structure" model).

Both predict per-object next-state delta (Δx, Δy, Δvx, Δvy).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class MonolithicWM(nn.Module):
    """Flat MLP. Input = pusher(2)+action(2)+ Nmax*(obj4+radius1). Output = Nmax*4 deltas."""
    def __init__(self, n_max=5, hidden=256):
        super().__init__()
        self.n_max = n_max
        in_dim = 4 + n_max * 5
        self.net = mlp([in_dim, hidden, hidden, hidden, n_max * 4])

    def forward(self, pusher, action, objects, radii):
        B, n, _ = objects.shape
        pad = self.n_max - n
        obj = torch.cat([objects, radii.unsqueeze(-1)], dim=-1)            # (B,n,5)
        if pad > 0:
            obj = torch.cat([obj, torch.zeros(B, pad, 5, device=obj.device)], dim=1)
        x = torch.cat([pusher, action, obj.reshape(B, -1)], dim=-1)
        out = self.net(x).reshape(B, self.n_max, 4)
        return out[:, :n]                                                  # (B,n,4) delta


class InteractionWM(nn.Module):
    """Object-centric interaction network: shared per-object + pairwise message functions on
    relative positions. Handles ANY number of objects with the same weights."""
    def __init__(self, hidden=128, emb=64):
        super().__init__()
        self.obj_enc = mlp([5, emb, emb])                 # per-object: [x,y,vx,vy,r]
        self.push_msg = mlp([emb + 2 + 4, hidden, emb])   # obj_emb + rel(obj->pusher) + action
        self.pair_msg = mlp([emb + emb + 2, hidden, emb])  # obj_i_emb + obj_j_emb + rel(i->j)
        self.dec = mlp([emb + emb + emb, hidden, hidden, 4])

    def forward(self, pusher, action, objects, radii):
        B, n, _ = objects.shape
        feat = torch.cat([objects, radii.unsqueeze(-1)], dim=-1)          # (B,n,5)
        h = self.obj_enc(feat)                                            # (B,n,emb)
        pos = objects[..., :2]                                            # (B,n,2)

        # message from pusher to each object (relative pos + action)
        rel_p = pos - pusher.unsqueeze(1)                                 # (B,n,2)
        act_b = action.unsqueeze(1).expand(B, n, 2)
        pad_act = torch.cat([act_b, torch.zeros(B, n, 2, device=act_b.device)], dim=-1)  # 4-dim
        m_push = self.push_msg(torch.cat([h, rel_p, pad_act], dim=-1))    # (B,n,emb)

        # pairwise messages: sum over j != i
        hi = h.unsqueeze(2).expand(B, n, n, h.shape[-1])
        hj = h.unsqueeze(1).expand(B, n, n, h.shape[-1])
        rel_ij = pos.unsqueeze(2) - pos.unsqueeze(1)                      # (B,n,n,2)
        pair = self.pair_msg(torch.cat([hi, hj, rel_ij], dim=-1))         # (B,n,n,emb)
        eye = torch.eye(n, device=pair.device).view(1, n, n, 1)
        m_pair = (pair * (1 - eye)).sum(dim=2)                            # (B,n,emb)

        return self.dec(torch.cat([h, m_push, m_pair], dim=-1))          # (B,n,4) delta
