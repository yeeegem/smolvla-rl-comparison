#!/usr/bin/env bash
# Score all three arms on the HELD-OUT seed range and render the comparison.
set -euo pipefail
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPISODES="${EPISODES:-100}"
SEEDS="${SEEDS:-0 1 2}"
BASE="${BASE:-checkpoints/base_smolvla}"
PPO="${PPO:-runs/ppo_seed0/checkpoints/last/pretrained_model}"
CRITIC="${CRITIC:-runs/gaf_critic}"

echo "=== base (frozen) ==="
uv run python -m grasprl.eval.evaluate --method base --checkpoint "$BASE" \
    --label base --episodes "$EPISODES" --seeds $SEEDS --split heldout

if [[ -d "$PPO" ]]; then
  echo "=== Flow-SDE PPO ==="
  uv run python -m grasprl.eval.evaluate --method ppo --checkpoint "$PPO" \
      --label ppo --episodes "$EPISODES" --seeds $SEEDS --split heldout
else
  echo "skipping PPO: $PPO not found (run scripts/train_ppo.sh)"
fi

if [[ -d "$CRITIC" ]]; then
  echo "=== Guided Action Flow (same frozen policy) ==="
  uv run python -m grasprl.eval.evaluate --method gaf --checkpoint "$BASE" \
      --critic "$CRITIC" --label gaf --episodes "$EPISODES" --seeds $SEEDS --split heldout
else
  echo "skipping GAF: $CRITIC not found (run scripts/train_gaf.sh)"
fi

uv run python -m grasprl.eval.report

# Figures, when the underlying runs exist.
[[ -f runs/ppo_seed0/metrics.jsonl ]] && \
  uv run python -m grasprl.eval.plots curve --metrics runs/ppo_seed0/metrics.jsonl
[[ -f results/gaf_sweep.json ]] && \
  uv run python -m grasprl.eval.plots sweep
true
