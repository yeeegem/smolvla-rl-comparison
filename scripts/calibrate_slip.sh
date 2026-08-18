#!/usr/bin/env bash
# Tune the sim's grasp physics until the FROZEN policy fails the way it really
# does, then freeze the result. Run this once, before either RL method: both are
# scored in this sim, so re-calibrating after seeing a result would invalidate
# the comparison.
set -euo pipefail
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python -m grasprl.sim.calibrate \
    --checkpoint "${CHECKPOINT:-checkpoints/base_smolvla}" \
    --episodes "${EPISODES:-40}" \
    --write "$@"
