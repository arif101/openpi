"""REASON-VLA v3 — refinement layer for foundation VLAs.

Components:
- physics_evaluator: forward-simulate action chunks in MuJoCo, compute cost
- (later) mppi: sampling-based refinement
- (later) hybrid: gradient + sampling composition

The atomic primitive everything builds on is ``PhysicsEvaluator``.
"""
