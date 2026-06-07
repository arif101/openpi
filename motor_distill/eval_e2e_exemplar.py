"""THE SOTA END-TO-END: exemplar (SAM-everything + DINOv2) BINDER -> wrist-cam MOTOR -> OFFICIAL BDDL success.
General/open-vocab: frozen SAM+DINOv2 visual correspondence to a per-object reference prototype bank (catalog
built from disjoint reference inits; product images in deployment). No training, no position memorization.

env @256 (motor untouched) + on-demand 1024 agentview render for binding; bind target ONCE pre-grasp, basket
ONCE at grasp. Bank built once over ALL objects (incl basket). Official success vs VLS 36.81% / pi0.5 23.69%.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_exemplar.py \
       --head data/motor_head.pt --bddl-dir <dir> --init-dir <dir> --proto-dir <std libero_object dir for catalog> \
       --proto-inits 0,2,4 --n 10 --trials 2 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from bind_exemplar import exemplar_bind, proto_crop
from dino_separability import dino_feat
from bind_foveate import nm
from train_wristcam_motor import WristMotor


def build_bank_all(proto_bddl_dir, proto_init_dir, proto_inits, R, cam, dev, half=60):
    """Prototype for EVERY object (incl basket) seen across the catalog scenes, from reference inits."""
    from libero.libero.envs import OffScreenRenderEnv
    acc = {}
    for bf in sorted(glob.glob(str(pathlib.Path(proto_bddl_dir) / "*.bddl"))):
        instr, objs, targets, distractors = parse_bddl(bf)
        stem = pathlib.Path(bf).stem
        fi = pathlib.Path(proto_init_dir) / f"{stem}.pruned_init"
        if not fi.exists(): continue
        inits = np.asarray(torch.load(fi, weights_only=False))
        for ti in proto_inits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R)
            env.seed(ti); env.reset(); obs = env.set_init_state(inits[ti]); sim = env.env.sim
            up = np.asarray(obs[cam + "_image"])[::-1].copy(); rb = resolve_bodies(sim, objs)
            for o in objs:
                if rb[o] is None: continue
                c = proto_crop(sim, up, R, cam, rb[o], half)
                if c is not None: acc.setdefault(nm(o), []).append(dino_feat(c, dev))
            env.close()
    bank = {}
    for k, fs in acc.items():
        v = np.mean(fs, 0); bank[k] = v / np.linalg.norm(v)
    return bank


def bind_hires(sim, name, all_names, bank, dev, res=1024):
    out = sim.render(width=res, height=res, camera_name="agentview", depth=True)
    rgb, dep = out if isinstance(out, tuple) else (out, None)
    if dep is None: return None
    g, nb = exemplar_bind(sim, np.asarray(rgb), np.asarray(dep)[..., None], name, all_names, bank,
                          "agentview", res, dev, vflip=True)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--proto-dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object")
    p.add_argument("--proto-init-dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_object")
    p.add_argument("--proto-inits", default="0,2,4")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=2)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--seed", type=int, default=20)
    p.add_argument("--bind-res", type=int, default=1024); p.add_argument("--img", type=int, default=128)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    proto_inits = [int(x) for x in args.proto_inits.split(",")]
    print("building prototype catalog (all objects)...", flush=True)
    bank = build_bank_all(args.proto_dir, args.proto_init_dir, proto_inits, args.bind_res, "agentview", dev)
    print(f"catalog: {sorted(bank)}", flush=True)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []; bound = 0; nb = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem; all_names = objs
        cont_bddl = next((o for o in objs if args.container in o), None)
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        s0 = args.init_start; nt = min(args.trials, (len(inits) - s0)) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=True)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, [T]); nb += 1
            goal_obj = bind_hires(sim, T, all_names, bank, dev, args.bind_res)
            if goal_obj is not None and rb[T] is not None and np.linalg.norm(goal_obj - body_pos(sim, rb[T])) < 0.06:
                bound += 1
            goal_cont = None; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            z0 = body_pos(sim, rb[T])[2] if rb[T] is not None else 0.0
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                if held and goal_cont is None and cont_bddl is not None:
                    goal_cont = bind_hires(sim, cont_bddl, all_names, bank, dev, args.bind_res)
                goal_w = goal_cont if (held and goal_cont is not None) else goal_obj
                goal_rel = (goal_w - ee).astype(np.float32) if goal_w is not None else np.zeros(3, np.float32)
                if chunk is None or ci >= args.replan:
                    wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img)
                    img = torch.tensor(np.transpose(wr.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    vec = np.concatenate([goal_rel, prop, [float(held)]]).astype(np.float32)
                    vec = ((vec - vm) / vs).astype(np.float32)
                    with torch.no_grad():
                        chunk = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]; ci = 0
                a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                close_cnt = close_cnt + 1 if grip > 0 else 0
                if (not held) and close_cnt > 8 and lifted > 0.02 and rb[T] is not None: held = True
                act = a[:7].copy(); act[6] = grip
                obs, _, done, _ = env.step(act.tolist())
                if rb[T] is not None: lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); env.close()
        print(f"  {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind={bound}/{nb}", flush=True)
    print(f"\n=== E2E EXEMPLAR (SAM+DINOv2 binder + wrist motor, OFFICIAL, {pathlib.Path(args.bddl_dir).name}): "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  |  target-bind<6cm={bound}/{nb} ===  [VLS 36.81% | pi0.5 23.69%]", flush=True)
    print("E2E_EXEMPLAR_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
