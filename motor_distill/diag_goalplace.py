"""Isolate WHY goal placement fails: teleport the target object to OUR computed place-aim point (no motor), settle
physics, check the env goal predicate. If success fires -> our aim is correct, the gap is motor/drop EXECUTION.
If not -> our place-aim (container localization: cont_w + aabb_top) is WRONG for goal receptacles (stove/cabinet/rack).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json \
  python3 motor_distill/diag_goalplace.py --bddl-dir <goal_swap> --init-dir <goal_swap> --n 10 --inits 20,21
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np, torch


def obj_free_joint(sim, tok):
    """Find the free-joint (7-qpos) whose body name contains `tok` (e.g. 'akita_black_bowl_1'); return qpos addr."""
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] != 0:   # 0 = free joint
            continue
        jbname = sim.model.body_id2name(sim.model.jnt_bodyid[j])
        if tok in jbname or jbname in tok:
            return int(sim.model.jnt_qposadr[j]), jbname
    return None, None


def region_pos(sim, target):
    """Position of the goal predicate's target REGION (e.g. wooden_cabinet_1_top_region / flat_stove_1_..._region).
    Try site, then body, then token-matched body (cabinet_top / burner_plate / rack). Returns (xyz, source)."""
    import numpy as np
    # 1) site
    try:
        sid = sim.model.site_name2id(target)
        return np.asarray(sim.data.site_xpos[sid], np.float32), f"site:{target}"
    except Exception: pass
    # 2) exact body
    try:
        bid = sim.model.body_name2id(target); return np.asarray(sim.data.body_xpos[bid], np.float32), f"body:{target}"
    except Exception: pass
    # 3) token-matched body: strip "_region", match a sub-body (e.g. cabinet_top, burner_plate, wine_rack)
    base = target.replace("_region", "")
    toks = set(base.split("_"))
    best = None; bn = 0
    for i in range(sim.model.nbody):
        nm = sim.model.body_id2name(i)
        if not nm: continue
        s = len(toks & set(nm.split("_")))
        if s > bn: bn, best = s, nm
    if best is not None:
        bid = sim.model.body_name2id(best); return np.asarray(sim.data.body_xpos[bid], np.float32), f"tokbody:{best}"
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", required=True)
    p.add_argument("--n", type=int, default=10); p.add_argument("--inits", default="20,21")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    from eval_physics_place import scene_bodies, resolve_noun, body_aabb_top, _body_tokens
    from language_planner import plan as lplan
    qinits = [int(x) for x in args.inits.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        stem = pathlib.Path(bf).stem
        gtxt = pathlib.Path(bf).read_text()
        goalstr = re.search(r"\(:goal(.*)", gtxt, re.S)
        goalline = " ".join(goalstr.group(1).split())[:120] if goalstr else "?"
        gpred = re.search(r"\((?:In|On)\s+(\S+)\s+(\S+?)\)", goalline)
        goal_region = gpred.group(2) if gpred else None
        subs = lplan(instr)
        pp = next((s for s in subs if s["skill"] == "place"), None)
        if pp is None:
            print(f"{stem[:40]:42s} ARTICULATED/no-place -> skip"); continue
        cands = scene_bodies(bf)
        To = resolve_noun(pp["obj"], cands); Co = resolve_noun(pp["target"], cands)
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        if not fi.exists(): continue
        inits = np.asarray(torch.load(fi, weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=128, camera_widths=128, camera_depths=False)
        print(f"\n{stem[:46]}\n  instr={instr!r}\n  goal=({goalline})  place obj={pp['obj']!r}->{To}  target={pp['target']!r}->{Co}", flush=True)
        hits = 0; ntr = 0
        for qi in qinits:
            if qi >= len(inits): continue
            env.seed(qi); env.reset(); env.set_init_state(inits[qi]); sim = env.env.sim
            rb = resolve_bodies(sim, [To, Co])
            cb = rb.get(Co)
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and Co and Co in b]
                cb = cand[0] if cand else None
            if rb.get(To) is None or cb is None:
                print(f"  init{qi}: unresolved To/cb"); continue
            contp = body_pos(sim, cb); top, halfxy = body_aabb_top(sim, sim.model.body_name2id(cb))
            # OUR current aim (container main-body center + its aabb top):
            aim_cont = np.array([contp[0], contp[1], top + 0.02], np.float32)
            # GOAL-REGION aim (the predicate's actual target sub-region):
            rp, rsrc = region_pos(sim, goal_region) if goal_region else (None, None)
            adr, jbn = obj_free_joint(sim, To)
            if adr is None:
                print(f"  init{qi}: no free joint for {To}"); continue
            res = {}
            for label, aim in [("CONT", aim_cont), ("REGION", (np.array([rp[0], rp[1], rp[2] + 0.04], np.float32) if rp is not None else None))]:
                if aim is None: res[label] = "n/a"; continue
                env.set_init_state(inits[qi]); sim = env.env.sim   # reset pose each trial
                sim.data.qpos[adr:adr+3] = aim; sim.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
                sim.forward()
                for _ in range(150): sim.step()
                try: ok = bool(env.env._check_success())
                except Exception as e: ok = f"err:{str(e)[:30]}"
                res[label] = ok
            hits += int(res.get("REGION") is True); ntr += 1
            print(f"  init{qi}: cont={contp.round(2)} aabbtop={top:.2f} | region={goal_region} src={rsrc} pos={rp.round(2) if rp is not None else None}"
                  f" | success CONT-aim={res['CONT']} REGION-aim={res['REGION']}", flush=True)
        print(f"  => teleport-to-aim success {hits}/{ntr}", flush=True)
        env.close()
    print("DIAGGOAL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
