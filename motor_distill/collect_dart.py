"""DART relabel collection (Laskey CoRL 2017) -- the dominant fix for BC covariate shift. Run pi0.5 on its
OWN competent task (pick the MEMORIZED object, place in basket -> pi0.5 binds CORRECTLY here), but INJECT
Gaussian noise into the EXECUTED action so the rollout visits drifted/off-path states. LOG pi0.5's (un-noised)
action as the CORRECTIVE LABEL at that drifted state. Training the student on these teaches it to recover from
drift (the states 40 clean demos never cover). Logged goal-relative + object-agnostic -> anti-memorization
preserved (labels are pi0.5's CORRECT-binding actions, stored relative to the goal). Same npz format as
collect_place so distill_motor_v2 trains on place+dart together. -> data/dart/*.npz

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/collect_dart.py --sigma 0.3
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--out", default="data/dart"); p.add_argument("--container", default="basket")
    p.add_argument("--n-scenes", type=int, default=10); p.add_argument("--k-init", type=int, default=4)
    p.add_argument("--horizon", type=int, default=260); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--sigma", type=float, default=0.3)        # action noise -> induces drift for DART
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    rng = np.random.default_rng(args.seed)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    kept = 0; tried = 0
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        if not distractors:
            continue
        M = distractors[0]                                    # pi0.5's competent/memorized target (binds correctly)
        for k in range(args.k_init):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
            env.seed(args.seed + k); env.reset(); obs = env.reset(); sim = env.env.sim
            try:
                rb = resolve_bodies(sim, [M, args.container + "_1"]); cb = rb[args.container + "_1"]
            except Exception:
                env.close(); continue
            tried += 1
            instr = f"pick up the {nm(M)} and place it in the {args.container}"
            z0 = body_pos(sim, rb[M])[2]
            EE, Q, G, TG, CG, A, PH = [], [], [], [], [], [], []
            chunk = None; ci = 0; lifted = 0.0
            for step in range(args.horizon):
                if chunk is None or ci >= args.replan:
                    o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                     obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                    chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                a_pi = chunk[ci].astype(np.float32)            # pi0.5's CORRECTIVE action = the LABEL
                mp = body_pos(sim, rb[M]).astype(np.float32); cp = body_pos(sim, cb).astype(np.float32)
                EE.append(np.asarray(obs["robot0_eef_pos"], np.float32)); Q.append(np.asarray(obs["robot0_eef_quat"], np.float32))
                G.append(np.asarray(obs["robot0_gripper_qpos"], np.float32)); TG.append(mp); CG.append(cp)
                A.append(a_pi[:7]); PH.append(1 if lifted > 0.03 else 0)
                # EXECUTE with injected noise on position+rot dims (not gripper) -> drift; pi0.5 corrects next step
                a_exec = a_pi.copy()
                a_exec[:6] = np.clip(a_exec[:6] + rng.normal(0, args.sigma, 6).astype(np.float32), -1, 1)
                obs, _, done, info = env.step(a_exec.tolist()); ci += 1
                lifted = max(lifted, body_pos(sim, rb[M])[2] - z0)
                if done:
                    break
            mp = body_pos(sim, rb[M]); cp = body_pos(sim, cb)
            # keep if pi0.5 recovered enough to make meaningful progress (lifted) -> useful corrective labels
            keep = bool(lifted > 0.04)
            env.close()
            if keep:
                np.savez(out / f"dart_{pathlib.Path(bf).stem[:18]}_{M}_{k}.npz",
                         ee=np.array(EE), quat=np.array(Q), grip=np.array(G), tgt=np.array(TG),
                         cont=np.array(CG), action=np.array(A), phase=np.array(PH))
                kept += 1
            print(f"  {pathlib.Path(bf).stem[:22]:24s} M={nm(M):13s} k={k} lifted={lifted*100:.0f}cm {'KEEP' if keep else 'drop'}", flush=True)
    print(f"\nKEPT {kept}/{tried} DART (noisy pi0.5-corrective) trajectories -> {out}", flush=True)
    print("COLLECT_DART_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
