"""Continuous blend test: a = alpha * a_pi05 + (1-alpha) * a_attractor.

Tests the core architecture idea (one continuous component, not a discrete
dispatch): does blending Pi0.5's expressive action expert with a goal-attractor
give a precision/robustness FRONTIER? Sweep a CONSTANT alpha 1->0:
  alpha=1 : pure Pi0.5 (precise in-dist 100%, collapses OOD 27%)
  alpha=0 : pure attractor (Exp 1 privileged: 70/63/30)
The question: is there a middle alpha that keeps pert0 ~100 AND lifts pert10 > 30?
If yes, the additive-blend architecture works and only the gating (alpha<-uncertainty)
remains. If no fixed alpha gives both, the composition is wrong.

Uses PRIVILEGED object pose for the attractor goal to ISOLATE the blend idea from
perception noise (a separate problem). Pi0.5 runs LIVE (policy.infer), closed-loop.
"""
from __future__ import annotations

import argparse
import glob
import math
import pathlib

import numpy as np

import rekey
from rekey import ee_in_object_frame, quat_normalize, quat_mul, quat_conj, quat_rotate
from eval_dmp_grasp import axisangle_from_quat, LIFT_M, GRASP_MAX_STEPS, GRIP_EPS, derive
from ndp import NDP, PlainHead

RESIZE = 224
LOOKAHEAD = 4
REPLAN = 5


def _quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def build_obs(obs, prompt):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(wr, RESIZE, RESIZE))
    state = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                            obs["robot0_gripper_qpos"]))
    return {"observation/image": img, "observation/wrist_image": wr,
            "observation/state": state, "prompt": str(prompt)}


def load_head(ckpt_path):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    T = ck["T"]
    net = NDP(cond_dim=3, n_dim=3, n_bf=20, T=T) if ck["kind"] == "ndp" else PlainHead(cond_dim=3, n_dim=3, T=T)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, np.asarray(ck["grasp_pos"], np.float32)


class Attractor:
    """Privileged-goal attractor controller; emits a 7-dim env action per step."""
    def __init__(self, net, grasp_pos, grasp_quat, apu, kp_rot=3.0):
        import torch
        self.torch = torch; self.net = net
        self.gp = torch.tensor(grasp_pos[None]); self.grasp_pos = grasp_pos
        self.grasp_quat = grasp_quat; self.apu = apu; self.kp_rot = kp_rot
        self.secured = 0

    def act(self, obs, sim, bid):
        torch = self.torch
        E = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs["robot0_eef_quat"], np.float32)
        Eq = np.array([eq[3], eq[0], eq[1], eq[2]], np.float32)
        O = sim.data.body_xpos[bid].astype(np.float32).copy()
        Oq = sim.data.body_xquat[bid].astype(np.float32).copy()
        e_rel, _ = ee_in_object_frame(E, Eq, O, Oq)
        dist = float(np.linalg.norm(e_rel - self.grasp_pos))
        if self.secured < 18:
            with torch.no_grad():
                c = torch.tensor(e_rel[None], dtype=torch.float32)
                traj = self.net(c, self.gp, c)[0].numpy()
            tgt = traj[min(LOOKAHEAD, len(traj) - 1)]
            desired = O + quat_rotate(Oq, tgt.astype(np.float32))
            a_pos = np.clip((desired - E) / self.apu, -1, 1)
            grip = 1.0 if dist < GRIP_EPS else -1.0
            self.secured = self.secured + 1 if dist < 0.012 else 0
        else:
            a_pos = np.array([0.0, 0.0, 1.0]); grip = 1.0
        tq = quat_mul(Oq, self.grasp_quat)
        a_rot = np.clip(self.kp_rot * axisangle_from_quat(quat_mul(tq, quat_conj(quat_normalize(Eq)))), -1, 1)
        return np.array([*a_pos, *a_rot, grip], np.float32)


def rollout(policy, net, grasp_pos, grasp_quat, apu, env, init, tname, prompt, alpha):
    env.reset(); obs = env.set_init_state(init)
    sim = env.env.sim; bid = int(sim.model.body_name2id(tname))
    z0 = float(sim.data.body_xpos[bid][2])
    attr = Attractor(net, grasp_pos, grasp_quat, apu)
    chunk = None; ci = 0
    for step in range(GRASP_MAX_STEPS):
        if alpha > 0 and (chunk is None or ci >= min(REPLAN, len(chunk))):
            chunk = np.asarray(policy.infer(build_obs(obs, prompt))["actions"], np.float32); ci = 0
        a_pi = chunk[ci] if alpha > 0 else np.zeros(7, np.float32); ci += 1
        a_at = attr.act(obs, sim, bid) if alpha < 1 else np.zeros(7, np.float32)
        a = alpha * a_pi + (1.0 - alpha) * a_at
        obs, _, done, _ = env.step(a.tolist())
        if float(sim.data.body_xpos[bid][2]) - z0 > LIFT_M:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)                       # NDP head
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--task-idx", type=int, default=3)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--ref-dir", default="data/keystone/pert0")
    p.add_argument("--perts", default="0,5,10")
    p.add_argument("--alphas", default="1.0,0.75,0.5,0.25,0.0")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    net, grasp_pos = load_head(args.ckpt)
    ref = sorted(glob.glob(str(pathlib.Path(args.ref_dir) /
                 f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    _, grasp_quat, tname, _, apu = derive(ref[:30], 16)
    prompt = str(np.load(ref[0], allow_pickle=True)["prompt"])

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)

    alphas = [float(a) for a in args.alphas.split(",")]
    perts = [int(x) for x in args.perts.split(",")]
    grid = {}
    for pert in perts:
        test = sorted(glob.glob(str(pathlib.Path(f"data/keystone/pert{pert}") /
                      f"PHYS_*_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))[: args.n]
        inits = [np.load(f, allow_pickle=True)["init_state_libero"] for f in test]
        for alpha in alphas:
            s = 0
            for init in inits:
                s += rollout(policy, net, grasp_pos, grasp_quat, apu, env, init, tname, prompt, alpha)
            grid[(pert, alpha)] = (s, len(inits))
            print(f"  pert{pert} alpha={alpha:.2f}: {s}/{len(inits)}={s/len(inits)*100:.0f}%", flush=True)

    print("\n=== BLEND FRONTIER: lift success % (rows alpha, cols pert) ===", flush=True)
    print("alpha\\pert  " + "  ".join(f"{pt:>5d}" for pt in perts), flush=True)
    for alpha in alphas:
        row = "  ".join(f"{grid[(pt,alpha)][0]/grid[(pt,alpha)][1]*100:4.0f}%" for pt in perts)
        print(f"  {alpha:.2f}     {row}", flush=True)
    print("\nWin = a row with pert0 ~>= 90 AND pert10 > 30 (beats both pure ends).", flush=True)
    print("BLEND_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
