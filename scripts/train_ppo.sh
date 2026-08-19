#!/usr/bin/env bash
# Arm A: Flow-SDE PPO. Updates 99.8M policy parameters.
# ~2.2 h per seed at the configs/ppo_flow_sde.yaml budget on a 16 GB laptop GPU.
set -euo pipefail
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHECKPOINT="${CHECKPOINT:-checkpoints/base_smolvla}"
# STEPS overrides configs/ppo_flow_sde.yaml without editing it, e.g.
#   STEPS=200000 bash scripts/train_ppo.sh 0
STEPS="${STEPS:-}"
EXTRA=()
[[ -n "$STEPS" ]] && EXTRA+=(--total-env-steps "$STEPS")

for seed in "${@:-0}"; do
  echo "=== PPO seed $seed ==="
  uv run python -m grasprl.ppo.train_ppo \
      --checkpoint "$CHECKPOINT" --out "runs/ppo_seed${seed}" --seed "$seed" \
      "${EXTRA[@]}"
done
