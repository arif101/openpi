"""SOTA path (inference-only, GENERAL): DINOv2 visual-exemplar BINDER picks which object is the named target ->
HIDE the other (distractor) objects (render-invisible; deployable = inpaint them out) -> pi0.5's OWN precise motor
grasps the relocated target on a CLEAN, in-distribution scene. Breakthrough: hiding distractors lifts pi0.5 swap
17%->63% (its 'memorization' was distractor confusion, not absolute). This uses pi0.5's near-perfect motor (no
goal-following cap, no <2cm localization, no LoRA). Official BDDL success on the SWAP axis vs VLS 36.81%.

Binder = DINOv2 prototype match among the scene's graspable objects (our validated 1.00-given-clean-crops result).
--binder oracle|dino : 'oracle' hides true distractors (ceiling of the HIDE mechanism); 'dino' uses the DINOv2
binder to choose the target (real, general).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_bind_hide.py \
       --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_swap --init-dir /root/LIBERO-Pro-data/init_files/libero_object_swap \
       --proto-dir /root/LIBERO-PRO/libero/libero/bddl_files/libero_object --proto-init-dir /root/LIBERO-PRO/libero/libero/init_files/libero_object \
       --n 10 --trials 3 --binder dino
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs
from bind_exemplar import build_bank, proto_crop
from dino_separability import dino_feat
from bind_foveate import nm


def geom_ids_for_body(sim, body_name):
    bid = sim.model.body_name2id(body_name)
    return [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero"); p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--proto-dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object")
    p.add_argument("--proto-init-dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_object")
    p.add_argument("--proto-inits", default="0,2,4"); p.add_argument("--bind-res", type=int, default=1024)
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--horizon", type=int, default=280)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--res", type=int, default=256)
    p.add_argument("--binder", default="dino", choices=["oracle", "dino"])
    args = p.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bank = None
    if args.binder == "dino":
        print("building DINOv2 prototype bank...", flush=True)
        bank = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                          args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], args.bind_res, "agentview", dev)
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []; bind_ok = 0; nb = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        graspables = [o for o in objs if args.container not in o]
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        nt = min(args.trials, len(inits)) if inits is not None else args.trials
        for t in range(nt):
            # hi-res pass to CHOOSE the target via DINOv2 (then a fresh env at motor res to run pi0.5)
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res)
            env.seed(t); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, graspables); nb += 1
            if args.binder == "oracle":
                chosen = T
            else:
                hi = sim.render(width=args.bind_res, height=args.bind_res, camera_name="agentview")   # hi-res for the binder
                up = np.asarray(hi)[::-1].copy()
                proto = bank.get(nm(T)); feats = {}
                for o in graspables:
                    if rb.get(o) is None: continue
                    c = proto_crop(sim, up, args.bind_res, "agentview", rb[o], 60)   # hi-res crop matching the 1024 bank
                    if c is not None: feats[o] = dino_feat(c, dev)
                chosen = max(feats, key=lambda o: float(feats[o] @ proto)) if (feats and proto is not None) else T
            bind_ok += int(chosen == T)
            # HIDE every graspable except the chosen target
            for o in graspables:
                if o == chosen or rb.get(o) is None: continue
                for g in geom_ids_for_body(sim, rb[o]): sim.model.geom_rgba[g, 3] = 0.0
            obs = env.set_init_state(inits[t]) if inits is not None else obs        # re-render with hides applied
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
        print(f"  {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind={bind_ok}/{nb}", flush=True)
    print(f"\n=== pi0.5 + {args.binder.upper()} BINDER + HIDE-distractors on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  | bind_ok={bind_ok}/{nb} ===  [swap baseline 17% | VLS 36.81%]", flush=True)
    print("BINDHIDE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
