"""Confirm the failure locus: give the exemplar binder PERFECT-RECALL proposals (every scene object's TRUE
center as a candidate crop), then DINOv2-match the TARGET prototype against all candidates. If disambiguation
is ~100% here, the deployed failure is PROPOSAL RECALL (-> build SAM-everything). If it still fails, the
prototype-vs-deployed-crop matching itself is the problem.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/priv_disambig.py \
       --res 1024 --proto-inits 0,2,4 --query-inits 20,22 --half 60
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos
from bind_exemplar import build_bank, proto_crop
from dino_separability import dino_feat
from bind_foveate import nm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object")
    p.add_argument("--init-dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_object")
    p.add_argument("--cam", default="agentview"); p.add_argument("--res", type=int, default=1024)
    p.add_argument("--proto-inits", default="0,2,4"); p.add_argument("--query-inits", default="20,22")
    p.add_argument("--half", type=int, default=60); p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"; R = args.res
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    proto_inits = [int(x) for x in args.proto_inits.split(",")]
    query_inits = [int(x) for x in args.query_inits.split(",")]
    print("building bank...", flush=True)
    bank = build_bank(bddls, args.init_dir, proto_inits, R, args.cam, dev, args.half)
    right = 0; total = 0; conf = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        cand_objs = [o for o in objs if "basket" not in o]            # graspable candidates
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{stem}.pruned_init", weights_only=False))
        proto = bank.get(nm(T))
        if proto is None: continue
        for ti in query_inits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R)
            env.seed(ti); env.reset(); obs = env.set_init_state(inits[ti]); sim = env.env.sim
            up = np.asarray(obs[args.cam + "_image"])[::-1].copy(); rb = resolve_bodies(sim, cand_objs)
            feats, names = [], []
            for o in cand_objs:
                if rb[o] is None: continue
                c = proto_crop(sim, up, R, args.cam, rb[o], args.half)
                if c is not None: feats.append(dino_feat(c, dev)); names.append(o)
            if not feats: env.close(); continue
            sims = np.stack(feats) @ proto; pick = names[int(sims.argmax())]
            total += 1; ok = (pick == T); right += int(ok)
            if not ok: conf.append((nm(T), nm(pick)))
            print(f"  {stem[:24]:26s} i{ti} pick='{nm(pick)}' {'OK' if ok else 'WRONG'}", flush=True)
            env.close()
    print(f"\n=== PERFECT-RECALL disambiguation (target proto vs all-object true crops): {right}/{total} = "
          f"{right/max(total,1):.2f} ===  confusions={conf}", flush=True)
    print("PRIV_DISAMBIG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
