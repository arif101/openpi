"""PhysVLA model components.

Architecture (PyTorch — JAX integration deferred to phase 2):

  observation (image + proprio) ─┐
                                 ├─► encoder ─► z_t ∈ ℝ^256
  proprioception (qpos/qvel/ee) ─┘                  │
                                                    ├─► dynamics(z_t, a_t) ─► z_{t+1}
                                                    └─► aux heads (training-only)
                                                          ├─ wrench_head(z)  → 6-DoF EE wrench
                                                          ├─ pose_head(z)    → N_obj × 7 (xyz + quat)
                                                          └─ slip_head(z)    → 3-DoF EE/object rel-vel

Training objective:
  L = w_dyn * ||z_{t+1} - encoder(obs_{t+1})||²            (self-consistency)
    + w_w   * MSE(wrench_head(z_t),  true_wrench)
    + w_p   * MSE(pose_head(z_t),    true_object_poses)
    + w_s   * MSE(slip_head(z_t),    true_slip_velocity)

The aux heads are the trick that forces the latent to encode physics. Once
the encoder has been trained, the heads can be discarded — the physics
knowledge lives in the latent and is consumed by `dynamics` at inference.

Param budget at default sizes:
  image encoder (resnet-ish):    ~3M
  proprio encoder:                 ~50k
  latent fusion MLP:              ~100k
  dynamics transformer (4 layer):  ~1M
  aux heads (MLPs):               ~200k
  Total:                          ~4M

Tiny by VLA standards. We can scale up if it works.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass
class PhysVLAConfig:
    """All architectural hyperparameters in one place."""

    # Latent dim. 256 is enough to encode physics state for ~10 objects.
    latent_dim: int = 256

    # Image encoder.
    image_size: int = 224
    image_channels: int = 3
    image_emb_dim: int = 256

    # Proprio. Pi0.5/LIBERO obs: qpos (26) + qvel (24) + ee_pose (3+4) + gripper (2) = 59.
    # Default to over-provisioned and let real input set the in_features.
    proprio_dim: int = 59
    proprio_emb_dim: int = 128

    # Action.
    action_dim: int = 7

    # Dynamics transformer.
    dyn_n_layers: int = 4
    dyn_n_heads: int = 4
    dyn_ff_dim: int = 1024

    # Aux head config.
    # Number of tracked objects per task (max). LIBERO ~8 per task.
    n_objects: int = 12
    # Slip velocity is 3-D (linear velocity of object in EE frame).
    slip_dim: int = 3
    # Wrench is 6-D (force + torque).
    wrench_dim: int = 6


class ImageEncoder(nn.Module):
    """Small CNN → flat embedding. Roughly resnet-shape but ~3M params.

    Replace with a ViT or a Pi0.5 backbone in phase 2 once the thesis is
    validated. For now this is fast to train and easy to debug.
    """

    def __init__(self, cfg: PhysVLAConfig):
        super().__init__()
        # 224x224 → 14x14 by 4 stride-2 convs. Channel ramp 32→64→128→256.
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.image_channels, 32, 7, stride=2, padding=3),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
        )
        # Adaptive pool → fixed-size embedding regardless of input size.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, cfg.image_emb_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: [B, 3, H, W] in [0, 1]
        x = self.stem(img)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class ProprioEncoder(nn.Module):
    """Concatenated proprio state → embedding. Pure MLP."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LatentEncoder(nn.Module):
    """Fuses image + proprio embeddings into the physics-aware latent z."""

    def __init__(self, cfg: PhysVLAConfig, proprio_in_dim: int):
        super().__init__()
        self.cfg = cfg
        self.image_enc = ImageEncoder(cfg)
        self.proprio_enc = ProprioEncoder(proprio_in_dim, cfg.proprio_emb_dim)
        in_dim = cfg.image_emb_dim + cfg.proprio_emb_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Linear(512, cfg.latent_dim),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        im = self.image_enc(image)
        pr = self.proprio_enc(proprio)
        return self.fuse(torch.cat([im, pr], dim=-1))


class DynamicsModel(nn.Module):
    """Forward dynamics in latent space: (z_t, a_t) -> z_{t+1}.

    Implemented as a small Transformer over [z_t, a_t]. The transformer is
    overkill for a single-step prediction but trivially extends to multi-step
    rollouts when we need them.
    """

    def __init__(self, cfg: PhysVLAConfig):
        super().__init__()
        self.cfg = cfg
        self.action_emb = nn.Linear(cfg.action_dim, cfg.latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.dyn_n_heads,
            dim_feedforward=cfg.dyn_ff_dim,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=cfg.dyn_n_layers)
        self.head = nn.Linear(cfg.latent_dim, cfg.latent_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # z: [B, D], a: [B, A]
        a_emb = self.action_emb(a)
        seq = torch.stack([z, a_emb], dim=1)  # [B, 2, D]
        out = self.tr(seq)
        return z + self.head(out[:, 0])  # residual update to z


class AuxHeads(nn.Module):
    """Physics-supervised heads. Used only during training.

    Each head is a small MLP that takes the latent and predicts a continuous
    physics quantity. Training-time MSE losses force the latent to encode
    enough physics for these predictions to be accurate. At inference these
    are discarded — the dynamics model uses the physics-rich latent
    implicitly.
    """

    def __init__(self, cfg: PhysVLAConfig):
        super().__init__()
        self.cfg = cfg

        def head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(cfg.latent_dim, 256),
                nn.LayerNorm(256), nn.ReLU(inplace=True),
                nn.Linear(256, out_dim),
            )

        # EE 6-DoF wrench
        self.wrench = head(cfg.wrench_dim)
        # Per-object pose: xyz (3) + quat (4) = 7 per object, max n_objects
        self.pose = head(cfg.n_objects * 7)
        # Slip velocity (linear, 3-D)
        self.slip = head(cfg.slip_dim)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "wrench": self.wrench(z),
            "pose": self.pose(z),  # [B, n_objects*7] — reshape in loss to [B, n_objects, 7]
            "slip": self.slip(z),
        }


class PhysVLAModel(nn.Module):
    """The full PhysVLA forward stack: encoder + dynamics + aux heads."""

    def __init__(self, cfg: PhysVLAConfig, proprio_in_dim: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = LatentEncoder(cfg, proprio_in_dim)
        self.dynamics = DynamicsModel(cfg)
        self.aux = AuxHeads(cfg)

    def encode(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.encoder(image, proprio)

    def step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(z, action)

    def aux_predictions(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.aux(z)


def physvla_loss(
    model: PhysVLAModel,
    batch: dict[str, torch.Tensor],
    *,
    w_dyn: float = 1.0,
    w_wrench: float = 2.0,
    w_pose: float = 1.0,
    w_slip: float = 1.0,
    contact_emphasis: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Future-step aux supervision: aux heads predict t+1 physics from the
    DYNAMICS-OUTPUT latent, not current physics from the encoded current state.

    This fixes the tautology problem of v1: in v1 the encoder saw current
    object_pos in proprio, then the pose head read it back from z_t. The
    aux loss was trivially zero. In v2:

        z_t      = encode(obs_t)
        z_tp1    = dynamics(z_t, a_t)
        wrench_pred = wrench_head(z_tp1)   # predicts t+1 wrench
        pose_pred   = pose_head(z_tp1)     # predicts t+1 pose
        slip_pred   = slip_head(z_tp1)     # predicts t+1 slip

    Now the dynamics module MUST produce a latent that encodes future physics
    to satisfy the loss. The encoder + dynamics together are forced to
    represent the physical state in a way that the heads can decode.

    Also: contact_emphasis weights the wrench loss by 1+α·||wrench_target||,
    so contact moments (where wrench is large) contribute more than the
    common no-contact baseline (where wrench≈0).

    batch contains:
      image_t:    [B, 3, H, W]
      proprio_t:  [B, P]
      action_t:   [B, A]
      image_tp1:  [B, 3, H, W]
      proprio_tp1:[B, P]
      wrench_tp1: [B, 6]              <-- TARGETS AT t+1 NOW
      pose_tp1:   [B, n_objects, 7]
      slip_tp1:   [B, 3]
    """
    z_t = model.encode(batch["image_t"], batch["proprio_t"])
    z_tp1_pred = model.step(z_t, batch["action_t"])
    z_tp1_target = model.encode(batch["image_tp1"], batch["proprio_tp1"])
    L_dyn = nn.functional.mse_loss(z_tp1_pred, z_tp1_target.detach())

    # Aux heads consume the DYNAMICS-OUTPUT latent.
    preds = model.aux_predictions(z_tp1_pred)

    # Wrench at t+1, with contact-emphasis weighting.
    wrench_target = batch["wrench_tp1"]
    wrench_mag = torch.linalg.norm(wrench_target, dim=-1, keepdim=True)
    sample_weights = 1.0 + contact_emphasis * wrench_mag
    L_wrench = (((preds["wrench"] - wrench_target) ** 2) * sample_weights).mean()

    # Pose at t+1.
    pose_target_flat = batch["pose_tp1"].reshape(batch["pose_tp1"].shape[0], -1)
    expected = model.cfg.n_objects * 7
    if pose_target_flat.shape[-1] < expected:
        pad = torch.zeros(
            pose_target_flat.shape[0], expected - pose_target_flat.shape[-1],
            device=pose_target_flat.device, dtype=pose_target_flat.dtype,
        )
        pose_target_flat = torch.cat([pose_target_flat, pad], dim=-1)
    else:
        pose_target_flat = pose_target_flat[:, :expected]
    L_pose = nn.functional.mse_loss(preds["pose"], pose_target_flat)

    # Slip at t+1.
    L_slip = nn.functional.mse_loss(preds["slip"], batch["slip_tp1"])

    total = w_dyn * L_dyn + w_wrench * L_wrench + w_pose * L_pose + w_slip * L_slip
    return total, {
        "L_total":  float(total.detach()),
        "L_dyn":    float(L_dyn.detach()),
        "L_wrench": float(L_wrench.detach()),
        "L_pose":   float(L_pose.detach()),
        "L_slip":   float(L_slip.detach()),
    }
