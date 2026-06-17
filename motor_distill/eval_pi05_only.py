"""Plain frozen-pi0.5 eval on a LIBERO-PRO axis dir (our own measured baseline for suites the verb-router sends to
pi0.5, e.g. libero_goal_*). Same protocol as our cascade evals (n tasks x trials, init-start offset)."""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--config-name", default="pi05_libero"); p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=1)
    p.add_argument("--horizon", type=int, default=520); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    succ = []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        stem = pathlib.Path(bf).stem
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res, camera_depths=False)
            env.seed(args.seed + ti); env.reset()
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            chunk = None; ci = 0
            for step in range(args.horizon):
                if chunk is None or ci >= args.replan:
                    o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                     obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                    chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); env.close()
        print(f"  {stem[:40]:42s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
    print(f"\n=== PI05-ONLY on {pathlib.Path(args.bddl_dir).name}: {np.mean(succ):.3f} ({sum(succ)}/{len(succ)}) ===", flush=True)
    print("PI05ONLY_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
