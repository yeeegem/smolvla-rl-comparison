#!/usr/bin/env bash
# Copy the frozen SmolVLA checkpoint that both RL methods start from.
#
# This is the sim+real co-trained policy from sim2real-soarm-benchmark: 53%
# success on the real SO-ARM101 with 30% of trials lost to grasp slip. Every
# number in results/comparison.md is measured against it, so it is vendored into
# this repo rather than referenced across repos -- if the source run is retrained
# or deleted, this comparison must not silently change meaning.
set -euo pipefail

# Machine-specific default: the sibling checkout this checkpoint was trained in
# (github.com/yeeegem/sim2real-soarm-benchmark). Override SRC to point at your
# own copy of runs/smolvla_cotrain/checkpoints/020000/pretrained_model.
SRC="${SRC:-../sim2real-soarm-benchmark/runs/smolvla_cotrain/checkpoints/020000/pretrained_model}"
DST="${DST:-checkpoints/base_smolvla}"

if [[ ! -d "$SRC" ]]; then
  echo "source checkpoint not found: $SRC" >&2
  echo "set SRC=/path/to/pretrained_model and re-run" >&2
  exit 1
fi

mkdir -p "$(dirname "$DST")"
rm -rf "$DST"
cp -r "$SRC" "$DST"
chmod -R u+rw "$DST"

cat > "$DST/PROVENANCE.md" <<EOF
# Base checkpoint provenance

| | |
|---|---|
| source | \`$SRC\` |
| copied | $(date -u +%Y-%m-%dT%H:%M:%SZ) |
| policy | SmolVLA, fine-tuned from \`lerobot/smolvla_base\` |
| dataset | \`cotrain/sim_real\` -- 1000 MuJoCo episodes + 100 real episodes oversampled 3x (1300 eps / 272,451 frames) |
| training | 20,000 steps, batch 32, lr 1e-4, \`train_expert_only\`, frozen vision encoder |
| chunk | \`chunk_size 50\`, \`n_action_steps 50\`, \`num_steps 10\` flow-matching steps |
| I/O | state (6) + front/wrist 480x640 -> action (6), padded to \`max_action_dim 32\` |
| task string | "Pick up a red cube and put it in the blue cup" |

## Measured real-arm baseline (30 tier-A trials)

| | |
|---|---|
| success | 16/30 = 53% |
| grasp_slip | 9/30 = 30% of trials, 64% of failures |
| grabbed_nothing | 4/30 = 13% of trials |
| collision/unsafe | 1/30 = 3% of trials |
| mode balance \`|P(left)-0.5|\` | 0.50 (collapsed to the right cube) |

Grasp slip is the target of this repo: counting slips as successes would put the
same policy at 83%.
EOF

du -sh "$DST"
echo "wrote $DST/PROVENANCE.md"
