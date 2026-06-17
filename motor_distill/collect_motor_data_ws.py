"""POSITION-DIVERSE data engine for the factored wrist-cam motor (Phase 2).

Our wrist motor's inputs are ONLY {wrist-cam, relative-goal(3), proprio, held} -- NO absolute coords, NO global
frame -- so it is ARCHITECTURALLY INCAPABLE of memorizing positions. Its 50% swap cap is therefore a DATA-COVERAGE
gap: it only ever saw standard positions, so the wrist view is OOD at relocated (swap) positions. Fix = give it
expert demos AT relocated positions. But pi0.5 ALONE fails at swap (~17%) -> too few demos. So we turn the Phase-1
finding into a DATA ENGINE: HIDE the distractor objects (alpha=0 -> clean in-distribution scene) so pi0.5 recovers
to ~50-63% at swap, and log its successful chunks as distillation samples. pi0.5 is used ONLY offline here; the
deployed motor is entirely ours (and the deploy pipeline also hides, so train==test).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/collect_motor_data_hide.py \
       --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_swap \
       --init-dir /root/LIBERO-Pro-data/init_files/libero_object_swap \
       --out data/motor_demos_swap --trials 8 --seeds 0,1 --container basket
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs, _quat2axisangle
from collect_motor_data import surface_point


def geom_ids_for_body(sim, body_name):
    bid = sim.model.body_name2id(body_name)
    return [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--host", default="localhost"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--out", default="data/motor_demos_swap"); p.add_argument("--container", default="basket")
    p.add_argument("--trials", type=int, default=8); p.add_argument("--seeds", default="0,1")
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--img", type=int, default=128); p.add_argument("--hide", type=int, default=1)
    p.add_argument("--goal-mode", default="surface", choices=["surface", "body", "depthflip"])   # depthflip = honest localizer (co-adaptation: train==test)
    p.add_argument("--cont-depth", type=int, default=0)   # depth-detect container TOP z (de-hardcode z_cont_center 0.035; general basket->plate); MATCH the --cont-depth deploy
    args = p.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _wcp
    policy = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))
    print(f"{len(bddls)} tasks, seeds={seeds}, trials={args.trials}, hide={args.hide}", flush=True)
    n_ok = 0; n_try = 0; n_samp = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        stem = pathlib.Path(bf).stem
        # instruction-derived pick target (BDDL targets[0] can be the CONTAINER on some suites, e.g. spatial plates)
        import re as _re
        T = targets[0]
        m = _re.search(r"pick_up_the_(.+)$", stem)
        if m:
            toks = m.group(1).split("_"); cut = len(toks)
            for stop in ("between", "next", "on", "from", "in", "and"):
                if stop in toks: cut = min(cut, toks.index(stop))
            tc = "_".join(toks[:cut])
            cand = next((o for o in targets if tc and tc in o and args.container not in o), None) or \
                   next((o for o in objs if tc and tc in o and args.container not in o), None)
            if cand is not None: T = cand
        cont_name = args.container + "_1"
        if args.container == "auto":   # GOAL suite: per-task container from (:goal (On X Y)); skip articulated/push
            _gm = _re.search(r"\(:goal.*?\((?:On|In)\s+(\w+)\s+(\w+)\)", pathlib.Path(bf).read_text(), _re.S)
            if _gm is None or _gm.group(2).startswith("main_table"):
                print(f"  {stem[:30]} skip-articulated/push", flush=True); continue
            T = _gm.group(1); cont_name = _re.sub(r"(_[a-z]+)+_region$", "", _gm.group(2))
        # SPATIAL relational which-instance: 2 identical bowls disambiguated by relation -> label T correctly
        # (else targets[0]/stem picks an arbitrary bowl while pi0.5 grasps the relationally-correct one -> mislabeled data).
        _relrel = None; _relnoun = None
        if args.container != "auto":
            from language_planner import plan as _lp
            for _s in _lp(instr):
                if _s["skill"] == "grasp" and _s.get("relation") and not str(_s["relation"]).startswith("instance-"):
                    _relrel = _s["relation"]; _relnoun = _s["obj"]; break
        distract_objs = [o for o in objs if o != T and o != cont_name and args.container not in o]
        T_base = T; distract_base = list(distract_objs)   # immutable base; relational T re-resolved per-trial
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        for seed in seeds:
            nt = min(args.trials, len(inits)) if inits is not None else args.trials
            for t in range(nt):
                n_try += 1
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=True)
                env.seed(seed + t); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
                T = T_base; distract_objs = list(distract_base)   # reset from immutable base each trial
                if _relrel:   # relational which-instance (needs sim, per-init): pick the correct bowl + recompute distractors
                    from eval_physics_place import select_instance as _si, scene_bodies as _sb
                    _ch = _si(sim, _relnoun, _relrel, _sb(bf))
                    if _ch:
                        T = _ch; distract_objs = [o for o in objs if o != T and o != cont_name and args.container not in o]
                rb = resolve_bodies(sim, [T, cont_name] + distract_objs); cb = rb.get(cont_name)
                if cb is None:
                    cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and cont_name in b]
                    cb = cand[0] if cand else None
                if rb[T] is None or cb is None: env.close(); continue
                if args.hide:
                    for o in distract_objs:
                        if rb.get(o) is None: continue
                        for g in geom_ids_for_body(sim, rb[o]):
                            sim.model.geom_rgba[g, 3] = 0.0          # render-invisible (physics unchanged)
                    obs = env.set_init_state(inits[t]) if inits is not None else obs   # re-render with alpha applied
                # TWO FIXED-WORLD goal points (pre-grasp graspable point + container drop point), localized ONCE at
                # the start (= what perception gives at deploy). NO state machine: the policy reads grasp PHASE from the
                # wrist image and learns grasp->transport->release + gripper timing from the teacher's actions itself.
                if args.goal_mode == "body":
                    obj_sp = body_pos(sim, rb[T]).astype(np.float32); cont_sp = body_pos(sim, cb).astype(np.float32)
                elif args.goal_mode == "depthflip":
                    # CO-ADAPTATION: log the SAME honest localizer outputs the deployed pipeline sees (train==test):
                    # object via depth-flip surface localize (~2cm), container via ray-plane at center height (no parallax)
                    import robosuite.utils.camera_utils as _cu
                    from eval_physics_place import depth_localize as _dl, obj_pixel as _op, ray_plane as _rp
                    _dm = _cu.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]))[::-1].copy()
                    obj_sp = _dl(sim, _op(sim, rb[T], 256), _dm, 256, mode="object")
                    if obj_sp is None: obj_sp = body_pos(sim, rb[T]).astype(np.float32)
                    if args.cont_depth:   # de-hardcode container z: depth-detect TOP (basket rim / plate top), match deploy
                        _cd = _dl(sim, _op(sim, cb, 256), _dm, 256, mode="container")
                        cont_sp = _rp(sim, cb, float(_cd[2]), 256) if _cd is not None else _rp(sim, cb, 0.035, 256)
                    else:
                        cont_sp = _rp(sim, cb, 0.035, 256)
                else:
                    obj_sp = surface_point(sim, rb[T], np.asarray(obs["agentview_depth"]), 256)
                    cont_sp = surface_point(sim, cb, np.asarray(obs["agentview_depth"]), 256)
                WR, OBJ, CON, PR, CH = [], [], [], [], []
                chunk = None; ci = 0
                for step in range(args.horizon):
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                    if chunk is None or ci >= args.replan:
                        wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                        o_in = build_obs(np.asarray(obs["agentview_image"]), wr_raw,
                                         obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                        chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                        wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img)
                        prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                               np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                        WR.append(image_tools.convert_to_uint8(wr))
                        OBJ.append((obj_sp.astype(np.float32) - ee)); CON.append((cont_sp.astype(np.float32) - ee))
                        PR.append(prop); CH.append(chunk.copy())
                    a = chunk[ci]; ci += 1
                    obs, _, done, _ = env.step(a[:7].tolist())
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                env.close()
                if ok and len(WR) > 3:
                    n_ok += 1; n_samp += len(WR)
                    np.savez_compressed(out / f"{stem[:55]}_s{seed}_t{t}.npz",
                                        wrist=np.asarray(WR, np.uint8), obj_rel=np.asarray(OBJ, np.float32),
                                        cont_rel=np.asarray(CON, np.float32), proprio=np.asarray(PR, np.float32),
                                        chunk=np.asarray(CH, np.float32))
                print(f"  {stem[:34]:36s} s{seed} t{t} {'OK' if ok else '..'} samp={len(WR)} "
                      f"(ok {n_ok}/{n_try}, {n_samp} samp)", flush=True)
    print(f"\n=== COLLECTED {n_ok}/{n_try} successful swap rollouts ({100*n_ok/max(n_try,1):.0f}%), "
          f"{n_samp} distill samples -> {out} ===", flush=True)
    print("COLLECTHIDE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
