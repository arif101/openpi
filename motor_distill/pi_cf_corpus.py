"""Counterfactual-invariance CORPUS generator (brick #1, the frontier experiment).
For each LIBERO scene it produces the three ingredients the 3-arm ablation needs, using pi0.5's own competent
rollouts (relabel = CAST-style) + sim ground-truth edits:

  (A) BC demos  : roll out pi0.5 on the object it BINDS correctly (the memorized M = distractors[0]); store
                  (agentview224, wrist224, state, action) RELABELED with instruction "pick up the {M} ...".
                  M varies across scenes -> covers many referents. Shared by Arm-2 (BC) and Arm-3 (ours).
  (B) Invariance pairs : at sampled timesteps, also render a DISTRACTOR-EDITED twin (teleport a non-M, non-basket
                  object to a new on-table spot) -> (obs, obs_twin) with SAME instruction. Arm-3 enforces
                  action INVARIANCE across the pair.
  (C) Sensitivity refs : store, per timestep, the 3D positions of M (the referent) and M2 (another present
                  object) + the alt instruction "pick up the {M2} ...". Arm-3 enforces action SENSITIVITY:
                  the chunk must diverge toward each named referent.  (3D from sim GT now; foveation later.)

-> npz per (scene, seed): images_main[T,224,224,3], images_wrist[T,...], state[T,8], action[T,7], instr (M),
   instr_alt (M2), tgt_ref[T,3] (M xyz), tgt_alt[T,3] (M2 xyz), twin_main[T,...] (distractor-edited), has_twin[T].

Run: PYTHONPATH=third_party/libero:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi_cf_corpus.py --out data/cf_corpus
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def free_joint_qposadr(sim, prefix):
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] != 0:
            continue
        bn = sim.model.body_id2name(int(sim.model.jnt_bodyid[j]))
        if bn and prefix in bn:
            return int(sim.model.jnt_qposadr[j])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--out", default="data/cf_corpus"); p.add_argument("--container", default="basket")
    p.add_argument("--n-scenes", type=int, default=10); p.add_argument("--k-init", type=int, default=4)
    p.add_argument("--horizon", type=int, default=260); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--twin-every", type=int, default=10); p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    rng = np.random.default_rng(args.seed)
    R = lambda im: image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(im[::-1, ::-1]), 224, 224))

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    kept = 0
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        if not distractors:
            continue
        M = distractors[0]                                            # pi0.5's correctly-bound object
        others = [o for o in (list(targets) + list(distractors)) if o != M]
        for k in range(args.k_init):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
            env.seed(args.seed + k); env.reset(); obs = env.reset(); sim = env.env.sim
            try:
                rb = resolve_bodies(sim, [M] + others + [args.container + "_1"]); cb = rb[args.container + "_1"]
            except Exception:
                env.close(); continue
            M2 = next((o for o in others if o in rb), None)            # an alternative present referent
            instr = f"pick up the {nm(M)} and place it in the {args.container}"
            instr_alt = f"pick up the {nm(M2)} and place it in the {args.container}" if M2 else instr
            z0 = body_pos(sim, rb[M])[2]; chunk = None; ci = 0; lifted = 0.0
            IM, WR, ST, AC, TR, TA, TW, HT = [], [], [], [], [], [], [], []
            for step in range(args.horizon):
                if chunk is None or ci >= args.replan:
                    o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                     obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                    chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                a = chunk[ci].astype(np.float32); ci += 1
                IM.append(R(np.asarray(obs["agentview_image"]))); WR.append(R(np.asarray(obs["robot0_eye_in_hand_image"])))
                st = np.concatenate([np.asarray(obs["robot0_eef_pos"], np.float32),
                                     np.asarray(obs["robot0_eef_quat"], np.float32),
                                     np.asarray(obs["robot0_gripper_qpos"], np.float32)]).astype(np.float32)
                ST.append(st); AC.append(a[:7])
                TR.append(body_pos(sim, rb[M]).astype(np.float32))
                TA.append(body_pos(sim, rb[M2]).astype(np.float32) if M2 else body_pos(sim, rb[M]).astype(np.float32))
                # (B) invariance twin: teleport a distractor far on-table, render, restore
                if M2 and (step % args.twin_every == 0):
                    adr = free_joint_qposadr(sim, M2.replace(" ", "_"))
                    if adr is not None:
                        saved = sim.data.qpos[adr:adr + 3].copy()
                        sim.data.qpos[adr:adr + 3] = saved + np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), 0.0])
                        sim.forward()
                        twin = R(np.asarray(sim.render(camera_name="agentview", width=256, height=256)[::-1])) \
                            if False else R(np.asarray(env.env._get_observations()["agentview_image"]))
                        sim.data.qpos[adr:adr + 3] = saved; sim.forward()
                        TW.append(twin); HT.append(1)
                    else:
                        TW.append(IM[-1]); HT.append(0)
                else:
                    TW.append(IM[-1]); HT.append(0)
                obs, _, done, _ = env.step(a[:7].tolist())
                lifted = max(lifted, body_pos(sim, rb[M])[2] - z0)
                if done:
                    break
            keep = lifted > 0.04
            env.close()
            if keep:
                np.savez_compressed(out / f"cf_{pathlib.Path(bf).stem[:16]}_{M}_{k}.npz",
                    images_main=np.array(IM, np.uint8), images_wrist=np.array(WR, np.uint8), state=np.array(ST),
                    action=np.array(AC), instr=instr, instr_alt=instr_alt, tgt_ref=np.array(TR), tgt_alt=np.array(TA),
                    twin_main=np.array(TW, np.uint8), has_twin=np.array(HT, np.int32))
                kept += 1
            print(f"  {pathlib.Path(bf).stem[:18]:20s} M={nm(M):13s} M2={nm(M2) if M2 else '-':12s} k={k} "
                  f"lifted={lifted*100:3.0f}cm T={len(AC)} twins={int(np.sum(HT))} {'KEEP' if keep else 'drop'}", flush=True)
    print(f"\nKEPT {kept} cf-corpus episodes -> {out}", flush=True)
    print("CF_CORPUS_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
