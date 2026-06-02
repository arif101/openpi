"""Exp 3: closed-loop grasp with the NDP head driven by PERCEPTION (not privileged
state). Each rollout estimates the object position from the live cameras via
Pi0.5 frozen features -> perception head P (world encoder), feeds it to the NDP
attractor. Object ORIENTATION is unperturbed in these position-only experiments,
so we use a fixed canonical quat from demos (not privileged).

Reports NDP+perception-lift vs Pi0.5-lift for pert 0/5/10. The real program test:
does perception (~1.7cm median, 3.6cm mean) + the closed-loop attractor still
re-target a moved object and beat Pi0.5's lookup-table collapse?

Object pos is static during approach, so we re-estimate every RE_EST steps (cheap).
First step prints estimated vs privileged O as a sanity guard.
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import jax.numpy as jnp
import numpy as np
import torch

import rekey
from rekey import ee_in_object_frame, quat_normalize, quat_mul, quat_conj, quat_rotate
from eval_dmp_grasp import pi05_lifted, axisangle_from_quat, LIFT_M, GRASP_MAX_STEPS, GRIP_EPS, derive
from ndp import NDP, PlainHead

from openpi.contact_mpc.features.extractor import extract_spatial_features_from_dict
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import download
import jax

torch.set_num_threads(8)
LOOKAHEAD = 4
RE_EST = 25
N_BASE = 196


class PooledHead(torch.nn.Module):
    def __init__(self, D):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.SiLU(),
                                       torch.nn.Linear(256, 64), torch.nn.SiLU(),
                                       torch.nn.Linear(64, 3))

    def forward(self, X):
        return self.mlp(X.mean(1))


def load_pi05(ckpt):
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10,
                               paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    params = _model.restore_params(download.maybe_download(ckpt), dtype=jnp.bfloat16)
    m = cfg.load(params); m.eval()
    return m


def resize224(imgs):
    imgs = imgs[:, ::-1, ::-1, :]
    out = jax.image.resize(imgs.astype(np.float32), (imgs.shape[0], 224, 224, 3), "bilinear")
    return np.asarray(jnp.clip(out, 0, 255).astype(jnp.uint8))


def perceive(model, P, base_img, wrist_img):
    base = resize224(base_img[None]); wr = resize224(wrist_img[None])
    data = {
        "image": {"base_0_rgb": base, "left_wrist_0_rgb": wr, "right_wrist_0_rgb": np.zeros_like(base)},
        "image_mask": {k: np.ones(1, bool) for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        "state": np.zeros((1, 32), np.float32),
    }
    grid, _ = extract_spatial_features_from_dict(model, data)
    with torch.no_grad():
        return P(torch.tensor(grid[:, :N_BASE]))[0].numpy().astype(np.float32)


def load_head(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    T = ck["T"]
    net = NDP(cond_dim=3, n_dim=3, n_bf=20, T=T) if ck["kind"] == "ndp" else PlainHead(cond_dim=3, n_dim=3, T=T)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck["kind"], np.asarray(ck["grasp_pos"], np.float32)


def rollout(net, P, model, grasp_pos, grasp_quat, canon_Oq, env, init, tname, apu, verbose, kp_rot=3.0):
    env.reset(); obs = env.set_init_state(init)
    sim = env.env.sim; bid = int(sim.model.body_name2id(tname))
    z0 = float(sim.data.body_xpos[bid][2])
    gp = torch.tensor(grasp_pos[None]); secured = 0
    O = None; Oq = canon_Oq
    for step in range(GRASP_MAX_STEPS):
        if O is None or step % RE_EST == 0:
            O = perceive(model, P, np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]))
            if verbose and step == 0:
                Opriv = sim.data.body_xpos[bid].astype(np.float32)
                print(f"      O_est={O.round(3)} O_priv={Opriv.round(3)} err={np.linalg.norm(O-Opriv)*100:.1f}cm", flush=True)
        E = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs["robot0_eef_quat"], np.float32)
        Eq = np.array([eq[3], eq[0], eq[1], eq[2]], np.float32)
        e_rel, _ = ee_in_object_frame(E, Eq, O, Oq)
        dist = float(np.linalg.norm(e_rel - grasp_pos))
        if secured < 18:
            with torch.no_grad():
                c = torch.tensor(e_rel[None], dtype=torch.float32)
                traj = net(c, gp, c)[0].numpy()
            tgt_obj = traj[min(LOOKAHEAD, len(traj) - 1)]
            desired_world = O + quat_rotate(Oq, tgt_obj.astype(np.float32))
            a_pos = np.clip((desired_world - E) / apu, -1, 1)
            grip = 1.0 if dist < GRIP_EPS else -1.0
            secured = secured + 1 if dist < 0.012 else 0
        else:
            a_pos = np.array([0.0, 0.0, 1.0]); grip = 1.0
        tq = quat_mul(Oq, grasp_quat)
        a_rot = np.clip(kp_rot * axisangle_from_quat(quat_mul(tq, quat_conj(quat_normalize(Eq)))), -1, 1)
        obs, _, done, _ = env.step([*a_pos, *a_rot, grip])
        if float(sim.data.body_xpos[bid][2]) - z0 > LIFT_M:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)                      # NDP head
    p.add_argument("--percep", default="data/keystone/heads/percep_t3.pt")
    p.add_argument("--pi05", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--pert-dir", required=True)
    p.add_argument("--ref-dir", default="data/keystone/pert0")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    net, kind, grasp_pos = load_head(args.ckpt)
    pk = torch.load(args.percep, map_location="cpu", weights_only=False)
    P = PooledHead(pk["D"]); P.load_state_dict(pk["state_dict"]); P.eval()
    model = load_pi05(args.pi05)

    ref = sorted(glob.glob(str(pathlib.Path(args.ref_dir) /
                 f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    _, grasp_quat, tname, _, apu = derive(ref[:30], 16)
    # canonical object orientation from demos (position-only perturbation -> orientation is fixed)
    oqs = [np.load(f, allow_pickle=True)["object_quat"] for f in ref[:30]]
    names = list(np.load(ref[0], allow_pickle=True)["object_names"])
    ti = rekey.select_target_object(np.load(ref[0], allow_pickle=True)["object_pos"], names)
    canon_Oq = quat_normalize(np.mean([o[0, ti] for o in oqs], 0).astype(np.float32))

    test = sorted(glob.glob(str(pathlib.Path(args.pert_dir) /
                  f"PHYS_*_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))[: args.n]
    pi05b = sum(pi05_lifted(f, tname) for f in test)

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    head = 0
    for i, f in enumerate(test):
        init = np.load(f, allow_pickle=True)["init_state_libero"]
        ok = rollout(net, P, model, grasp_pos, grasp_quat, canon_Oq, env, init, tname, apu, verbose=(i < 3))
        head += ok
        print(f"  [{i+1}/{len(test)}] {'LIFT' if ok else '----'} (percep+{kind} {head}/{i+1})", flush=True)
    n = len(test); pert = args.pert_dir.split("pert")[-1]
    print(f"\nEXP3 [percep+{kind}] task {args.task_idx} pert={pert}: {head}/{n}={head/n*100:.1f}%  "
          f"vs Pi0.5 {pi05b}/{n}={pi05b/n*100:.1f}%", flush=True)
    print("EXP3_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
