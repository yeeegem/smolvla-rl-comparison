#!/usr/bin/env bash
# Pick the guidance hyperparameters on the VALIDATION seed range only.
#
# The paper this reproduces found a +10.0 pp validation gain that was worth only
# +2.5 pp on a locked held-out set. Tuning here and reporting on held-out seeds
# in eval_all.sh is what keeps this repo from making the same claim.
set -euo pipefail
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python -m grasprl.gaf.sweep \
    --critic "${CRITIC:-runs/gaf_critic}" \
    --checkpoint "${CHECKPOINT:-checkpoints/base_smolvla}" \
    --episodes "${EPISODES:-50}" "$@"
