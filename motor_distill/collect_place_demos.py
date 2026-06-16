"""Convert LIBERO expert demos -> POSITION-INVARIANT relative-goal grasp+place supervision.

Expert demos are at ORIGINAL positions; we replay each demo's recorded sim states to read the
object + container poses, compute obj_rel/cont_rel (relative to EE = position-invariant), and pair
them with the demo's wrist cam + proprio + action chunks. Output npz matches the coadapt format so
train_wristcam_motor.py trains a clean grasp+place primitive from expert (not pi0.5/scripted) data.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl python3 motor_distill/collect_place_demos.py \
       --out data/place_demos --max-demos 50
"""
from __future__ import annotations
import argparse, glob, pathlib, re, os
import numpy as np, h5py
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", default="/root/LIBERO-PRO/libero/datasets/libero_object")
    ap.add_argument("--bddl-dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object")
    ap.add_argument("--out", default="data/place_demos")
    ap.add_argument("--container", default="basket")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--max-demos", type=int, default=50)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--wrist-flip", type=int, default=1)   # match eval preprocessing ([::-1,::-1]); A/B if motor fails
    ap.add_argument("--tasks", type=int, default=99)
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    os.makedirs(args.out, exist_ok=True)
    demo_files = sorted(glob.glob(f"{args.demo_dir}/*.hdf5"))[: args.tasks]
    tot = 0
    for df in demo_files:
        stem = pathlib.Path(df).stem.replace("_demo", "")
        bf = f"{args.bddl_dir}/{stem}.bddl"
        if not pathlib.Path(bf).exists():
            print("NO BDDL for", stem, flush=True); continue
        instr, objs, targets, distractors = parse_bddl(bf)
        graspables = [o for o in objs if args.container not in o]
        T = targets[0] if targets else None
        m = re.match(r"pick_up_the_(.+?)_and", stem)
        if m:
            tc = m.group(1); cand = next((o for o in graspables if tc in o), None)
            if cand: T = cand
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=128, camera_widths=128)
        env.seed(0); env.reset(); sim = env.env.sim
        rb = resolve_bodies(sim, graspables + [args.container + "_1"]); cb = rb.get(args.container + "_1")
        if rb.get(T) is None or cb is None:
            print("BODY RESOLVE FAIL", stem, "T=", T, flush=True); env.close(); continue
        f = h5py.File(df, "r"); data = f["data"]; nd = 0
        for dk in list(data.keys())[: args.max_demos]:
            g = data[dk]
            states = np.asarray(g["states"]); acts = np.asarray(g["actions"]).astype(np.float32)
            Tt = states.shape[0]
            obj_rel = np.zeros((Tt, 3), np.float32); cont_rel = np.zeros((Tt, 3), np.float32); objz = np.zeros(Tt, np.float32)
            proprio = np.zeros((Tt, 5), np.float32); wr_list = []
            for t in range(Tt):
                sim.set_state_from_flattened(states[t]); sim.forward()
                # LIVE re-render obs to EXACTLY match the eval convention (demo-stored eye_in_hand_rgb is a
                # different convention than the live robot0_eye_in_hand_image the eval feeds -> was the bug).
                obs = env.env._get_observations()
                eef = np.asarray(obs["robot0_eef_pos"], np.float32)
                op = body_pos(sim, rb[T]).astype(np.float32); cp = body_pos(sim, cb).astype(np.float32)
                obj_rel[t] = op - eef; cont_rel[t] = cp - eef; objz[t] = op[2]
                proprio[t] = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                             np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                w = np.asarray(obs["robot0_eye_in_hand_image"])
                wr_list.append(image_tools.resize_with_pad(
                    np.ascontiguousarray(w[::-1, ::-1] if args.wrist_flip else w), args.img, args.img))
            held = (objz - objz[0] > 0.03).astype(np.int32)
            wr = np.stack(wr_list).astype(np.uint8)
            ch = np.zeros((Tt, args.chunk, 7), np.float32)
            for t in range(Tt):
                e = min(t + args.chunk, Tt); ch[t, :e - t] = acts[t:e]
                if e - t < args.chunk: ch[t, e - t:] = acts[Tt - 1]
            np.savez(f"{args.out}/{stem}__{dk}.npz", wrist=wr, obj_rel=obj_rel, cont_rel=cont_rel,
                     proprio=proprio, chunk=ch, held=held)
            nd += 1; tot += 1
        env.close(); f.close()
        print(f"{stem}: target={T} {nd} demos -> npz", flush=True)
    print(f"PLACE_DEMOS_DONE total={tot}", flush=True)


if __name__ == "__main__":
    main()
