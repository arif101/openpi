"""Diagnose the LEARNED grounding head's BOWL localization on spatial scenes (root-cause for grasp=0.00).
For each spatial task / a few inits: render agentview@gres, run the trained head with the bowl proto, take the
top-2 NMS peaks -> pixels (256 frame), and compare to the GT pixels of BOTH bowl instances (body_pos projection,
scoring ONLY). Prints peak pixels + min-distance to either bowl. If min-dist >> 18px -> head mislocalizes bowls.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json \
  python3 motor_distill/diag_relbind.py --bddl-dir <spatial_swap> --init-dir <spatial_swap> --ground-head data/ground_head_os.pt --n 6
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos
from bind_foveate import nm
from diag_dinodense import dino_dense
from train_ground_head import GroundHead


def peaks_px(gh, fg_t, proto, g, R, gres, k=2, nms=1, raw=False):
    with torch.no_grad():
        if raw:   # RAW dense cosine correspondence (no head) — robust on low-texture bowls
            pt = torch.tensor(np.asarray(proto), dtype=fg_t.dtype, device=fg_t.device)
            lo = (fg_t[0] * pt).sum(-1).clone()                  # (g,g)
        else:
            lo = gh(fg_t, torch.tensor(np.asarray(proto)[None]).to(fg_t.device))[0].reshape(g, g).clone()
    out = []
    for _ in range(k):
        pi = int(lo.argmax()); pr, pc = pi // g, pi % g
        r_up = (pr + 0.5) / g * gres; c = (pc + 0.5) / g * gres
        out.append((np.array([R - 1 - r_up / (gres / R), c / (gres / R)], np.float32), float(lo.max()), (pr, pc)))
        lo[max(0, pr - nms):pr + nms + 1, max(0, pc - nms):pc + nms + 1] = -1e9
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", required=True)
    p.add_argument("--ground-head", default="data/ground_head_os.pt"); p.add_argument("--n", type=int, default=6)
    p.add_argument("--inits", default="20,22,24")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import eval_physics_place as E
    gk = torch.load(args.ground_head, map_location=dev, weights_only=False)
    gh = GroundHead(gk["C"]).to(dev); gh.load_state_dict(gk["state"]); gh.eval()
    gres = gk["res"]; R = 256
    inits_q = [int(x) for x in args.inits.split(",")]
    print(f"protos available: {sorted(gk['protos'].keys())}", flush=True)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        stem = pathlib.Path(bf).stem
        bowls = [o for o in objs if "bowl" in o]
        if len(bowls) < 1: continue
        bnoun = nm(bowls[0])
        sp = E.build_scene_protos(bf, args.init_dir, dev, gk, (30, 32, 34))   # fresh per-scene proto
        proto = sp.get(bnoun)
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        if not fi.exists(): continue
        inits = np.asarray(torch.load(fi, weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
        print(f"\n{stem[:46]}  bowls={bowls}  proto[{bnoun!r}]={'OK' if proto is not None else 'MISSING'}", flush=True)
        for qi in inits_q:
            if qi >= len(inits): continue
            env.seed(qi); env.reset(); env.set_init_state(inits[qi]); sim = env.env.sim
            rb = resolve_bodies(sim, bowls)
            hi = np.asarray(sim.render(width=gres, height=gres, camera_name="agentview"))[::-1].copy()
            fg, g = dino_dense(hi, dev, gres); fg_t = torch.tensor(fg[None]).to(dev)
            class _A: pass
            _a = _A(); _a.res = R
            pk0 = (E._gh_peaks_px(gh, fg_t, proto, g, _a, gk, k=2, nms=2, raw=True, soft=True) if proto is not None else [])
            pk = [(px, 0.0, (0, 0)) for px in pk0]   # soft sub-patch peaks via the eval helper
            w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
            gts = {}
            for b in bowls:
                if rb.get(b) is None: continue
                px = cu.project_points_from_world_to_camera(body_pos(sim, rb[b])[None], w2p, R, R)[0]
                gts[b] = np.array([px[0], px[1]], np.float32)
            line = f"  init{qi}: GT " + " ".join(f"{b[-12:]}=({gts[b][0]:.0f},{gts[b][1]:.0f})" for b in gts)
            for j, (px, sc, patch) in enumerate(pk):
                dmin = min((float(np.hypot(px[0] - gts[b][0], px[1] - gts[b][1])) for b in gts), default=999)
                line += f"\n          peak{j} px=({px[0]:.0f},{px[1]:.0f}) patch={patch} score={sc:.2f} min-d-to-bowl={dmin:.1f}px {'HIT' if dmin < 18 else 'MISS'}"
            print(line, flush=True)
        env.close()
    print("RELBIND_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
