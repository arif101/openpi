"""Split the 10 NO_APPROACH traces by ee_to_bowl_min:
   close-but-no-grasp (8-12cm) vs way-off (>14cm).

For each subgroup compute:
  - At the moment of closest approach (ee_to_bowl_min_step), what was the
    EE doing? Velocity? Direction? Was it heading toward the bowl?
  - What's the actual ee_pos vs bowl_pos at that moment? Same z height?
    Same xy? Or wildly off?
  - In the way-off group: where is the EE during the rollout? Stuck at
    init, drifting elsewhere, or commanded to a wrong xy?
"""

import json
import pathlib

import numpy as np

results = json.loads(pathlib.Path("/tmp/openpi-data/in_false_diagnostic.json").read_text())
traces_dir = pathlib.Path("/tmp/openpi-data/recovery_source_traces")

no_approach = [r for r in results if r["submode"] == "NO_APPROACH"]
close = [r for r in no_approach if r["ee_to_bowl_min_cm"] < 12.0]
far = [r for r in no_approach if r["ee_to_bowl_min_cm"] >= 14.0]
print(f"NO_APPROACH total: {len(no_approach)}")
print(f"  close (8-12cm): {len(close)}")
print(f"  far (>=14cm):   {len(far)}")
print()

def inspect(r):
    d = np.load(traces_dir / r["trace"], allow_pickle=True)
    ee = d["ee_pos"]
    bowl = d["object_pos"][:, 0]
    act = d["action"]
    T = act.shape[0]
    t = r["ee_to_bowl_min_step"]
    # Relative position at closest approach
    rel = ee[t] - bowl[t]
    # EE velocity (3-step diff) at closest approach
    if 0 < t < T - 3:
        ee_vel = (ee[t + 3] - ee[t]) / 3
    else:
        ee_vel = np.zeros(3)
    # Action being commanded at closest approach
    act_at_t = act[t] if t < T else act[-1]
    # Was the EE heading toward the bowl right then?
    bowl_dir = bowl[t] - ee[t]
    bowl_dir_norm = bowl_dir / max(np.linalg.norm(bowl_dir), 1e-6)
    ee_vel_norm_mag = np.linalg.norm(ee_vel)
    dot_toward = float(np.dot(ee_vel / max(ee_vel_norm_mag, 1e-6), bowl_dir_norm)) if ee_vel_norm_mag > 1e-4 else 0.0
    return {
        "trace_short": r["trace"][:55],
        "ee_to_bowl_min_cm": r["ee_to_bowl_min_cm"],
        "step_at_min": t,
        "T": T,
        "ee_pos_at_min": ee[t].tolist(),
        "bowl_pos_at_min": bowl[t].tolist(),
        "rel_ee_minus_bowl_cm": (rel * 100).tolist(),
        "ee_vel_norm_mm_step": float(ee_vel_norm_mag * 1000),
        "ee_velocity_toward_bowl_dot": dot_toward,
        "action_dxyz_at_min": act_at_t[:3].tolist(),
        "action_gripper_at_min": float(act_at_t[6]),
    }

print("--- CLOSE subgroup (ee_to_bowl_min 8-12cm) ---")
print(f"{'trace':<60} {'min_cm':>6} {'step':>5} {'rel_xyz_cm':<22} {'ee_vel_mm':>10} {'->bowl':>8} {'act_dxyz':<22} {'grip':>6}")
print("-" * 145)
for r in close:
    info = inspect(r)
    rel_s = f"({info['rel_ee_minus_bowl_cm'][0]:+.1f},{info['rel_ee_minus_bowl_cm'][1]:+.1f},{info['rel_ee_minus_bowl_cm'][2]:+.1f})"
    act_s = f"({info['action_dxyz_at_min'][0]:+.2f},{info['action_dxyz_at_min'][1]:+.2f},{info['action_dxyz_at_min'][2]:+.2f})"
    print(f"{info['trace_short']:<60} {info['ee_to_bowl_min_cm']:>6.1f} {info['step_at_min']:>5} {rel_s:<22} "
          f"{info['ee_vel_norm_mm_step']:>9.2f} {info['ee_velocity_toward_bowl_dot']:>+8.2f} {act_s:<22} "
          f"{info['action_gripper_at_min']:>+6.2f}")

print()
print("--- FAR subgroup (ee_to_bowl_min >=14cm) ---")
print(f"{'trace':<60} {'min_cm':>6} {'step':>5} {'rel_xyz_cm':<22} {'ee_vel_mm':>10} {'->bowl':>8} {'act_dxyz':<22} {'grip':>6}")
print("-" * 145)
for r in far:
    info = inspect(r)
    rel_s = f"({info['rel_ee_minus_bowl_cm'][0]:+.1f},{info['rel_ee_minus_bowl_cm'][1]:+.1f},{info['rel_ee_minus_bowl_cm'][2]:+.1f})"
    act_s = f"({info['action_dxyz_at_min'][0]:+.2f},{info['action_dxyz_at_min'][1]:+.2f},{info['action_dxyz_at_min'][2]:+.2f})"
    print(f"{info['trace_short']:<60} {info['ee_to_bowl_min_cm']:>6.1f} {info['step_at_min']:>5} {rel_s:<22} "
          f"{info['ee_vel_norm_mm_step']:>9.2f} {info['ee_velocity_toward_bowl_dot']:>+8.2f} {act_s:<22} "
          f"{info['action_gripper_at_min']:>+6.2f}")

# Summary stats
print()
def summary(label, group):
    if not group: return
    print(f"\n{label} subgroup (N={len(group)}):")
    rels = [inspect(r)["rel_ee_minus_bowl_cm"] for r in group]
    rels = np.array(rels)
    print(f"  mean rel xyz (cm): ({rels[:,0].mean():+.1f}, {rels[:,1].mean():+.1f}, {rels[:,2].mean():+.1f})")
    print(f"  z-component mean:  {rels[:,2].mean():+.1f} cm  (positive = EE above bowl)")
    dots = [inspect(r)["ee_velocity_toward_bowl_dot"] for r in group]
    print(f"  mean cosine(velocity, toward-bowl): {np.mean(dots):+.2f}  (1=heading toward, -1=heading away)")

summary("CLOSE", close)
summary("FAR", far)
