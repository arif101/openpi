"""Eval the GOAL-CORRECTING mask-motor: condition on agentview + target MASK + wrist (NO point). The mask says
which object; the motor servos to it. On the SWAP axis (relocation), the mask is at the new position -> a
position-general motor grasps it BY CONSTRUCTION. Mask from instance-seg here (idealized binder; deployable = binder).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_maskmotor.py \
       --head data/maskmotor.pt --bddl-dir <dir> --init-dir <dir> --n 10 --trials 3 --init-start 0
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_maskmotor import MaskMotor


def tgt_mask(sim, body, seg_native, R, cam="agentview"):
    import robosuite.utils.camera_utils as cu
    from openpi_client import image_tools
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); px = cu.project_points_from_world_to_camera(body_pos(sim, body)[None], w2p, R, R)[0]
    r = int(min(max(px[0], 0), R - 1)); c = int(min(max(px[1], 0), R - 1))
    seg = seg_native.reshape(R, R); sid = seg[r, c]
    msk = (seg == sid).astype(np.uint8) * 255 if sid != 0 else np.zeros((R, R), np.uint8)
    return image_tools.resize_with_pad(np.ascontiguousarray(msk[::-1, ::-1, None].repeat(3, 2)), 128, 128)[..., 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/maskmotor.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--init-start", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--res", type=int, default=256)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"; R = args.res
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = MaskMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    pm, ps = ck["pm"], ck["ps"]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        s0 = args.init_start; nt = min(args.trials, (len(inits) - s0)) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True, camera_segmentations="instance")
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, [T, args.container + "_1"]); cb = rb[args.container + "_1"]
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]; cb = cand[0] if cand else None
            if rb[T] is None: env.close(); succ.append(0); continue
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            for step in range(args.horizon):
                if chunk is None or ci >= args.replan:
                    seg = np.asarray(obs["agentview_segmentation_instance"])
                    ag = image_tools.resize_with_pad(np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1]), 128, 128).astype(np.float32) / 255.0
                    mk = (tgt_mask(sim, cb if (held and cb is not None) else rb[T], seg, R).astype(np.float32) / 255.0)
                    agm = np.concatenate([np.transpose(ag, (2, 0, 1)), mk[None]], 0)
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]), 128, 128).astype(np.float32) / 255.0
                    pr = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])), np.asarray(obs["robot0_gripper_qpos"], np.float32), [float(held)])).astype(np.float32)
                    pr = ((pr - pm) / ps).astype(np.float32)
                    with torch.no_grad():
                        chunk = net(torch.tensor(agm)[None].to(dev), torch.tensor(np.transpose(wr, (2, 0, 1)))[None].to(dev), torch.tensor(pr)[None].to(dev)).cpu().numpy()[0]; ci = 0
                a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                close_cnt = close_cnt + 1 if grip > 0 else 0
                if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                act = a[:7].copy(); act[6] = grip
                obs, _, done, _ = env.step(act.tolist())
                lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); env.close()
        print(f"  {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
    print(f"\n=== MASK-MOTOR (agentview+mask+wrist, NO point) on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)}) ===  [goal-correcting test]", flush=True)
    print("MASKEVAL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
