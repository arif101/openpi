"""DAgger round (research's strongest next lever): roll out OUR STUDENT motor, and at each state the student
actually visits, query pi0.5 (the expert) for the correct action -> log (student-drifted-state, pi0.5-label).
This targets the student's REAL drift (carry slips), not random noise. Run in memorized-object scenes (pi0.5
binds correctly) with TRUE goals; store goal-relative -> anti-memorization preserved. Same npz format as
place/dart so distill_motor_v2 trains on place+dart+dagger together. -> data/dagger/*.npz

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/collect_dagger.py --head runs/motor_v2dart_head.pkl
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle, re
import numpy as np, jax, jax.numpy as jnp
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs
from distill_motor_v2 import motor_apply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--head", default="runs/motor_v2dart_head.pkl"); p.add_argument("--out", default="data/dagger")
    p.add_argument("--container", default="basket"); p.add_argument("--n-scenes", type=int, default=10)
    p.add_argument("--k-init", type=int, default=5); p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--ens-m", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=77)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    mh = jax.tree.map(jnp.asarray, pickle.load(open(args.head, "rb"))); K = mh["w3"].shape[1] // 7
    motor_fn = jax.jit(motor_apply)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    kept = 0; tried = 0
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        if not distractors:
            continue
        M = distractors[0]                                    # pi0.5 binds this correctly
        for k in range(args.k_init):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
            env.seed(args.seed + k); env.reset(); obs = env.reset(); sim = env.env.sim
            try:
                rb = resolve_bodies(sim, [M, args.container + "_1"]); cb = rb[args.container + "_1"]
            except Exception:
                env.close(); continue
            tried += 1
            instr = f"pick up the {nm(M)} and place it in the {args.container}"
            gobj0 = body_pos(sim, rb[M]).astype(np.float32); gcont = body_pos(sim, cb).astype(np.float32)
            z0 = gobj0[2]
            EE, Q, G, TG, CG, A, PH = [], [], [], [], [], [], []
            lifted = 0.0; held = False; close_cnt = 0; recent = []
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                gobj = body_pos(sim, rb[M]).astype(np.float32)   # current object pos (true goal)
                active = gcont if held else gobj
                # STUDENT action (temporal-ensembled relay motor)
                ch = np.asarray(motor_fn(mh, jnp.asarray(ee - active), jnp.asarray(np.asarray(obs["robot0_eef_quat"],np.float32)),
                                         jnp.asarray(np.asarray(obs["robot0_gripper_qpos"],np.float32))))
                recent.append((step, ch)); recent = recent[-K:]
                preds, ages = [], []
                for (s, c) in recent:
                    j = step - s
                    if 0 <= j < K: preds.append(c[j]); ages.append(j)
                wt = np.exp(-args.ens_m * np.array(ages)); wt /= wt.sum()
                sa = (np.stack(preds) * wt[:, None]).sum(0)
                grip = 1.0 if sa[6] > 0 else -1.0
                # EXPERT (pi0.5) label at this student-visited state
                o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                 obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                a_pi = np.asarray(policy.infer(o_in)["actions"], np.float32)[0][:7]
                EE.append(ee); Q.append(np.asarray(obs["robot0_eef_quat"],np.float32))
                G.append(np.asarray(obs["robot0_gripper_qpos"],np.float32)); TG.append(gobj); CG.append(gcont)
                A.append(a_pi); PH.append(1 if held else 0)     # LABEL = pi0.5's action at student state
                close_cnt = close_cnt + 1 if grip > 0 else 0
                if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                obs, _, done, _ = env.step(np.concatenate([sa[:6], [grip]]).tolist())   # EXECUTE STUDENT
                lifted = max(lifted, body_pos(sim, rb[M])[2] - z0)
                if done: break
            env.close()
            keep = len(A) > 30
            if keep:
                np.savez(out / f"dag_{pathlib.Path(bf).stem[:16]}_{M}_{k}.npz",
                         ee=np.array(EE), quat=np.array(Q), grip=np.array(G), tgt=np.array(TG),
                         cont=np.array(CG), action=np.array(A), phase=np.array(PH))
                kept += 1
            print(f"  {pathlib.Path(bf).stem[:20]:22s} M={nm(M):13s} k={k} student_lifted={lifted*100:.0f}cm steps={len(A)} {'KEEP' if keep else 'drop'}", flush=True)
    print(f"\nKEPT {kept}/{tried} DAgger (student-state, pi0.5-label) trajectories -> {out}", flush=True)
    print("COLLECT_DAGGER_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
