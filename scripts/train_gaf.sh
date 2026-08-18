#!/usr/bin/env bash
# Arm B: Guided Action Flow. Changes NO policy weights.
#   1. roll out the frozen policy          (~1 h)
#   2. fit the action-chunk critic ensemble (minutes)
#   3. pick beta/alpha/c on VALIDATION seeds (~30 min)
# The locked config is then reported once on held-out seeds by eval_all.sh.
set -euo pipefail
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHECKPOINT="${CHECKPOINT:-checkpoints/base_smolvla}"
EPISODES="${EPISODES:-600}"

uv run python -m grasprl.gaf.collect \
    --checkpoint "$CHECKPOINT" --episodes "$EPISODES" --out recordings/gaf_rollouts

uv run python -m grasprl.gaf.train_critic \
    --rollouts recordings/gaf_rollouts --out runs/gaf_critic

bash scripts/sweep_gaf.sh
