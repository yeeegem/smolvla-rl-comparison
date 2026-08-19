#!/usr/bin/env bash
# Choose the sim's grasp parameters and freeze them. Cells are scored on fitness
# as a TESTBED for comparing the two RL methods (headroom x sample yield), not on
# resemblance to the real arm, which is printed as reference only. Run this once,
# before either RL method: both are scored in this sim, so re-running it after
# seeing a result would invalidate the comparison.
set -euo pipefail
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python -m grasprl.sim.calibrate \
    --checkpoint "${CHECKPOINT:-checkpoints/base_smolvla}" \
    --episodes "${EPISODES:-40}" \
    --write "$@"
