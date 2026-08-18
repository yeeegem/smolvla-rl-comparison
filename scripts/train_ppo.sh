#!/usr/bin/env bash
# Arm A: Flow-SDE PPO. Updates 99.8M policy parameters.
# ~2.2 h per seed at the configs/ppo_flow_sde.yaml budget on a 16 GB laptop GPU.
set -euo pipefail
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHECKPOINT="${CHECKPOINT:-checkpoints/base_smolvla}"
for seed in "${@:-0}"; do
  echo "=== PPO seed $seed ==="
  uv run python -m grasprl.ppo.train_ppo \
      --checkpoint "$CHECKPOINT" --out "runs/ppo_seed${seed}" --seed "$seed"
done
