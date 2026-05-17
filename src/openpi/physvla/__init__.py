"""PhysVLA — physics-grounded VLA layer that bootstraps on Pi0.5.

The thesis: a learned forward-dynamics model whose latent is supervised to
encode continuous physics state (contact wrenches, object poses, slip
velocities). At inference, action selection is gradient-refined through this
dynamics model. The latent develops a physics 'instinct' the same way a
human's motor cortex does: by being forced to predict physics quantities
during development.

Layout:
    model.py         — latent encoder, forward dynamics, physics aux heads
    dataset.py       — torch Dataset over physvla_traces/*.npz
    (training loop in scripts/train_physvla_dynamics.py)

Phase 1: components below stand alone. Pi0.5 backbone is consulted only as
a baseline action proposer at inference time (not yet integrated into the
encoder). The latent dynamics + physics aux heads are pure-PhysVLA IP.
"""
