"""DAgger iteration for the wrist-cam motor: motor_v1 is position-robust (50% on swap w/ privileged goal), so let
IT generate demos AT the OOD positions (swap), keep the SUCCESSFUL episodes, and aggregate with the standard data.
This supplies the wrist-cam VIEW distribution the standard-only training was missing -> should lift swap toward 83%.

Logs the SAME schema as collect_motor_data.py (wrist,goal_rel,proprio,held,chunk) for successful episodes only.
Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/collect_dagger_wristcam.py \
       --head data/motor_head.pt --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_swap \
       --init-dir /root/LIBERO-Pro-data/init_files/libero_object_swap --out data/motor_demos_swap --trials 6 --seeds 0,1,2
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--out", default="data/motor_demos_swap")
    p.add_argument("--container", default="basket"); p.add_argument("--trials", type=int, default=6)
    p.add_argument("--seeds", default="0,1,2"); p.add_argument("--horizon", type=int, default=280)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--img", type=int, default=128)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))
    n_ok = 0; n_samp = 0
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
        for seed in seeds:
            nt = min(args.trials, len(inits)) if inits is not None else args.trials
            for t in range(nt):
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
                env.seed(seed + t); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
                rb = resolve_bodies(sim, [T, args.container + "_1"]); cb = rb[args.container + "_1"]
                if cb is None:
                    cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                    cb = cand[0] if cand else None
                if rb[T] is None or cb is None: env.close(); continue
                z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
                WR, GR, PR, HE, CH = [], [], [], [], []
                for step in range(args.horizon):
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                    active = body_pos(sim, cb) if held else body_pos(sim, rb[T])
                    goal_rel = (active.astype(np.float32) - ee)
                    if chunk is None or ci >= args.replan:
                        wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                        wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img)
                        img = torch.tensor(np.transpose(wr.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                        prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                               np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                        vec = np.concatenate([goal_rel, prop, [float(held)]]).astype(np.float32)
                        vecn = ((vec - vm) / vs).astype(np.float32)
                        with torch.no_grad():
                            chunk = net(img, torch.tensor(vecn)[None].to(dev)).cpu().numpy()[0]; ci = 0
                        WR.append(image_tools.convert_to_uint8(wr)); GR.append(goal_rel.copy())
                        PR.append(prop); HE.append(int(held)); CH.append(chunk.copy())
                    a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                    close_cnt = close_cnt + 1 if grip > 0 else 0
                    if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                    act = a[:7].copy(); act[6] = grip
                    obs, _, done, _ = env.step(act.tolist())
                    lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                env.close()
                if ok and len(WR) > 3:
                    n_ok += 1; n_samp += len(WR)
                    np.savez_compressed(out / f"dag_{stem[:26]}_s{seed}_t{t}.npz",
                                        wrist=np.asarray(WR, np.uint8), goal_rel=np.asarray(GR, np.float32),
                                        proprio=np.asarray(PR, np.float32), held=np.asarray(HE, np.int32),
                                        chunk=np.asarray(CH, np.float32))
                print(f"  {stem[:30]:32s} s{seed} t{t} {'OK' if ok else '..'} (kept {n_ok}/{n_samp}samp)", flush=True)
    print(f"\n=== DAGGER collected {n_ok} successful swap-position rollouts, {n_samp} samples -> {out} ===", flush=True)
    print("DAGGER_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
