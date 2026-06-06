"""DE-RISK the binding-fix decision: if T is UNAMBIGUOUS, does pi0.5 place it?
Physically teleport every object EXCEPT the counterfactual target T (and the basket) far below the floor, instruct
pi0.5 on T, score with the OFFICIAL bddl predicate (checks T). This removes the competing memorized object so the
only barrier left is whether pi0.5 CAN bind+place T at all.
  HIGH success -> binding is the sole barrier; a counterfactual-grounding LoRA will very likely hit SOTA (green-light).
  LOW success  -> the problem is deeper than binding; no steering/LoRA fixes it (don't invest).

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_isolate.py --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def teleport_away(sim, prefix):
    """Drop the free-joint object whose body name contains `prefix` far below the floor. Returns True if found."""
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] != 0:                       # 0 = free joint
            continue
        bname = sim.model.body_id2name(int(sim.model.jnt_bodyid[j]))
        if bname and prefix in bname:
            adr = int(sim.model.jnt_qposadr[j])
            sim.data.qpos[adr:adr+3] = [10.0, 10.0, -5.0]
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--horizon", type=int, default=300); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []
    for bf in bddls:
        instr_raw, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; remove = [o for o in (list(targets[1:]) + list(distractors))]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        nteleport = sum(int(teleport_away(sim, o)) for o in remove)
        sim.forward()
        for _ in range(10):                                  # let scene settle after teleport
            obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
        instr = f"pick up the {nm(T)} and place it in the {args.container}"
        try:
            rbT = resolve_bodies(sim, [T])[T]
        except Exception:
            rbT = None
        z0 = body_pos(sim, rbT)[2] if rbT else 0.0; lifted = 0.0; min_ee2T = 9.9
        chunk = None; ci = 0
        for step in range(args.horizon):
            if chunk is None or ci >= args.replan:
                o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                 obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
            obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
            if rbT:
                tp = body_pos(sim, rbT); ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                lifted = max(lifted, tp[2] - z0); min_ee2T = min(min_ee2T, float(np.linalg.norm(ee[:2] - tp[:2])))
            if done: break
        try: ok = bool(env.env._check_success())
        except Exception: ok = False
        succ.append(int(ok)); env.close()
        print(f"  {nm(T):14s} removed={nteleport}/{len(remove)} ee2T_min={min_ee2T*100:3.0f}cm liftT={lifted*100:3.0f}cm OFFICIAL={'Y' if ok else '.'}", flush=True)
    print(f"\n=== pi0.5 ISOLATED-T (distractors removed, N={len(succ)}, seed={args.seed}) place-T OFFICIAL: {np.mean(succ):.2f} ===", flush=True)
    print("PI05_ISOLATE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
