#!/usr/bin/env bash
# Record an annotated rollout of one arm, to see WHY it fails rather than how often.
# Overlays the commanded grip and the jaw gap it produces, distance from the grasp
# point to the nearer cube, whether both pads are loaded and with what force, and
# the cube height. Writes results/rollout_<method>.mp4.
set -euo pipefail
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
METHOD="${1:-base}"
shift || true
uv run python -m grasprl.eval.video --method "$METHOD" \
    --episodes "${EPISODES:-6}" --seed "${SEED:-0}" "$@"
