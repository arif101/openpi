"""Diagnose the SUPERVISED depth multi-instance head ON THE EVAL PATH (why eval bind 0.20 when train both-bowls=1.0).
Per spatial task/init: build the EVAL depth grid (eval_physics_place._depth_grid_eval), run InstHead top-2 peaks
(_inst_peaks_px), and report: (a) both-bowls recall in the eval path, (b) which bowl the relational resolve PICKS vs
the oracle-correct instance. Isolates detection (should be ~1.0) vs reference-localization vs resolve.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json \
  python3 motor_distill/diag_inst.py --bddl-dir <spatial_swap> --init-dir <spatial_swap> --inst-head data/inst_head_spatial.pt --ground-head data/ground_head_os.pt --n 6
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", required=True)
    p.add_argument("--inst-head", required=True); p.add_argument("--ground-head", default="data/ground_head_os.pt")
    p.add_argument("--n", type=int, default=6); p.add_argument("--inits", default="20,21,22")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    import eval_physics_place as E
    from train_inst_head import InstHead
    from train_ground_head import GroundHead
    from diag_dinodense import dino_dense
    from bind_foveate import nm
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ik = torch.load(args.inst_head, map_location=dev, weights_only=False)
    ih = InstHead(ik["C"]).to(dev); ih.load_state_dict(ik["state"]); ih.eval()
    gk = torch.load(args.ground_head, map_location=dev, weights_only=False)
    gres = ik["res"]; R = 256

    class A: pass
    a = A(); a.res = R; a.bind_res = 1024

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    qinits = [int(x) for x in args.inits.split(",")]
    tot_both = 0; n_both = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        stem = pathlib.Path(bf).stem
        bowls = [o for o in objs if "bowl" in o]
        if len(bowls) < 2: continue
        sp = E.build_scene_protos(bf, args.init_dir, dev, ik, (42, 44, 46))   # per-SCENE proto (fixes nm-collision)
        iproto = sp.get("akita black bowl")
        if iproto is None: iproto = ik["protos"].get(nm(bowls[0]))
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        if not fi.exists(): continue
        inits = np.asarray(torch.load(fi, weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
        print(f"\n{stem[:46]} bowls={bowls}", flush=True)
        for qi in qinits:
            if qi >= len(inits): continue
            env.seed(qi); env.reset(); obs = env.set_init_state(inits[qi]); sim = env.env.sim
            rb = resolve_bodies(sim, bowls)
            hi = np.asarray(sim.render(width=gres, height=gres, camera_name="agentview"))[::-1].copy()
            fg, g = dino_dense(hi, dev, gres); fg_t = torch.tensor(fg[None]).to(dev)
            depth_g = E._depth_grid_eval(sim, obs, g, gres)
            pk = E._inst_peaks_px(ih, fg_t, depth_g, iproto, g, a, ik, k=2, nms=2)
            w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
            gts = {b: cu.project_points_from_world_to_camera(body_pos(sim, rb[b])[None], w2p, R, R)[0] for b in rb if rb.get(b) is not None}
            line = f"  init{qi}: GT " + " ".join(f"{b[-1]}=({gts[b][0]:.0f},{gts[b][1]:.0f})" for b in gts)
            covered = 0
            for b in gts:
                d = min(float(np.hypot(pk[j][0] - gts[b][0], pk[j][1] - gts[b][1])) for j in range(len(pk)))
                covered += int(d < 18)
            both = int(covered == len(gts)); n_both += 1; tot_both += both
            for j, px in enumerate(pk):
                dmin = min(float(np.hypot(px[0] - gts[b][0], px[1] - gts[b][1])) for b in gts)
                line += f"\n      peak{j}=({px[0]:.0f},{px[1]:.0f}) min-d={dmin:.1f} {'HIT' if dmin<18 else 'MISS'}"
            line += f"   BOTH={'Y' if both else 'N'}"
            print(line, flush=True)
        env.close()
    print(f"\n=== EVAL-PATH both-bowls recall = {tot_both}/{n_both} = {tot_both/max(n_both,1):.3f} ===", flush=True)
    print("DIAGINST_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
