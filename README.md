# smolvla-grasp-rl-compare

**Two RL methods, one frozen policy, one failure mode.** A SmolVLA policy picks a red cube and
drops it in a blue cup on a real SO-ARM101. It succeeds 53% of the time, and its single biggest
failure is that the cube **slips out of the grip** -- 30% of all trials, 64% of all failures. Count
the slips as successes and the same policy is at 83%.

This repo asks which of two reinforcement-learning methods fixes that better, starting both from
the identical frozen checkpoint:

| | Arm A -- **Flow-SDE PPO** | Arm B -- **Guided Action Flow** |
|---|---|---|
| where the improvement lives | 99.8M updated policy parameters | a 4-layer MLP critic, policy untouched |
| what runs at inference | the fine-tuned policy | the **same frozen policy**, steered mid-denoise |
| training cost | hours of on-policy rollout | one offline rollout pass + minutes of regression |
| source | [`smolvla-ppo-cube-stacking`](../smolvla-ppo-cube-stacking) | [arXiv 2607.02092v1](https://arxiv.org/abs/2607.02092) |

Both are scored in the same simulator, with the same failure taxonomy the real arm is scored
with, and then on the real arm.

---

## Why this needed a new simulator

The checkpoint comes from `sim2real-soarm-benchmark`, which trained it on 1000 MuJoCo episodes
plus 100 real ones. That sim holds the cube with a **weld**: `Scene.attach()` fixes the cube to
the gripper with an `mjEQ_WELD` equality, and the whole arm is given collision mask 0 so nothing
ever touches anything. It is a good abstraction for recording demonstrations -- and it makes the
failure this repo studies **literally unrepresentable**. You cannot do RL on grasp slip in a
simulator where grasps cannot fail.

So the first thing here is a contact-physics grasp: two thin high-friction **pad geoms** on the
jaw faces, masked to collide only with cubes, leaving the rest of the arm non-colliding exactly as
before. Whether the cube stays in the gripper is then decided by friction, squeeze force and carry
dynamics.

The pads are anchored to numbers measured off the vendored MJCF, not guessed
(`scripts/tune_jaw_pads.py` re-derives and checks them):

```
fixed jaw inner face   x = -0.0079      moving jaw inner face  x = +0.0217
jaw gap at grip 19     29.64 mm   vs a 30 mm cube  ->  0.36 mm interference
jaw gap at grip 21     32.44 mm   ->  the cube is free
```

That last pair is the whole point. **The gripper channel decides hold-versus-drop within two
`RANGE_0_100` units**, and the jaws only reach the cube's top ~9 mm because the policy's grasp
reference sits at the fingertip. It is a precarious grasp, which is why the real arm drops it.

---

## What the sim can and cannot model

Building this turned up a result worth stating plainly, because it shapes every number the repo
produces.

**The frozen policy positions its jaws about ten times less accurately in MuJoCo than the weld
abstraction ever revealed.** Measured at the instant the gripper closes, offsets in the gripper
frame:

| | across the jaws (y) | grasp height (z) | tilt |
|---|---|---|---|
| scripted expert (succeeds) | 0.4 to 0.9 mm | +4.6 to +4.9 mm | 0° |
| frozen policy | **5.8 to 19.9 mm** | **+6.3 to +13.8 mm** | 0 to 4° |

A weld triggered by 35 mm proximity does not care. A 30 mm cube in a 29.6 mm jaw gap cares a
great deal: with a fully unassisted contact grasp the policy ends 70% of episodes in
`grabbed_nothing`, against 13% on the real arm. Pad width (12 up to 40 mm), pad height, sliding
friction (0.2 up to 3.0), servo torque (0.6 up to 3.35 Nm) and contact softness were each swept; none of
them moved it, because none of them is the problem. Meanwhile the *same* checkpoint scores **81%
in sim under the weld oracle**, with a median reach error of 1.2 cm -- its perception and
positioning transfer; what does not transfer is millimetre-scale placement.

So the two failure modes are separated, and only one of them is abstracted:

* **Acquisition** -- did the cube end up between the jaws. Abstracted. `Scene.capture()` cancels
  the *across-the-jaws* component of the placement error when the gripper closes within
  `grasp.capture_radius`, calibrated so the sim's grasp rate matches the real arm's 83%.
* **Retention** -- did it stay there. **Real physics**, untouched. Grasp height, cube yaw,
  gripper command and carry acceleration all still decide the outcome, and those are exactly the
  variables a policy can learn to fix.

`Scene.capture` corrects one axis and one axis only; it does not weld, and it does not tidy up the
grasp. Close high on the cube and the jaws still catch its top edge. Meet a corner instead of a
face and it still squirts out. Command the gripper two units too open and it still drops.

**This is the load-bearing assumption of the sim half of the comparison, and it is why the real-arm
runs are not optional.** Set `grasp.capture_radius: null` for the unassisted grasp, or
`grasp.mode: weld` for the original abstraction, and re-run any result you want to check.

---

## Status

Everything below runs, and 49 tests pass. **The calibration gate has not been passed yet** and
that is the next thing to do.

| | state |
|---|---|
| contact grasp, pads, masks, weld fallback | built, tested, verified against the MJCF |
| env, failure taxonomy, potential-based reward | built, tested |
| Arm A: Flow-SDE PPO | built; trains end to end at ~70 ticks/s (a 400k-tick run is ~1.6 h) |
| Arm B: Guided Action Flow | built; collect to critic to guided sampling verified end to end |
| sim evaluator, report, plots, real-arm harness | built |
| **calibration to the real 53/30/13 profile** | **not yet achieved** |

Best 20-episode probe so far, against a target of 53% success / 30% slip / 13% grabbed-nothing:

```
capture_radius  torque |  succ  lift  slip  noth
        0.020    1.40  |  0.25  0.45  0.20  0.55
```

Acquisition is the gap: the sim grasps about 50% of the time against the real arm's 83%, so
`grabbed_nothing` is roughly four times too high, and the RL methods would be scored partly on a
failure mode that is a sim artifact. `scripts/calibrate_slip.sh` sweeps a wider grid with 40
episodes per cell and **refuses to write a configuration that does not get close**, so run it and
read the gate before starting either RL method. If the gate cannot be met, widening
`--capture` past 0.045 or raising `grasp.pad_half_width` are the levers with the most headroom,
and `grasp.mode: weld` is always available as the upper-bound ablation.

## Quickstart

```bash
uv sync --extra dev
scripts/fetch_base_checkpoint.sh              # copies the frozen SmolVLA checkpoint

MUJOCO_GL=egl uv run --extra dev pytest tests/ -q
MUJOCO_GL=egl uv run python scripts/tune_jaw_pads.py --out scene_views/grasp.png
```

### 1. Calibrate the sim to the real failure profile -- do this first

Both methods are scored in this simulator, so it has to fail the way the real arm fails before
either of them runs. Re-calibrating afterwards would invalidate the comparison.

```bash
bash scripts/calibrate_slip.sh                # ~40 min; writes configs/scene.yaml
```

Sweeps `capture_radius` (moves `grabbed_nothing`) against `gripper_forcerange` (moves
`grasp_slip`), scores each cell by L1 distance to the real arm's measured
**53% success / 30% slip / 13% grabbed-nothing**, and **refuses to write** a cell that does not
get close. If the gate fails, stop and fix the sim rather than running RL in it.

### 2. Arm A -- Flow-SDE PPO (~2.2 h)

```bash
bash scripts/train_ppo.sh 0                   # or: seeds 0 1 2
```

### 3. Arm B -- Guided Action Flow (~2 h)

```bash
bash scripts/train_gaf.sh                     # collect -> fit critic -> tune on validation seeds
```

### 4. Head-to-head

```bash
bash scripts/eval_all.sh                      # held-out seeds, all three arms -> results/comparison.md
```

### 5. On the real SO-ARM101 (~20 trials per arm)

Edit `configs/eval_real.yaml` for your port and cameras. All three arms run through one control
loop at the same decision cadence as the sim, so the comparison is like-for-like.

```bash
uv run python -m grasprl.real.run --method base \
    --checkpoint checkpoints/base_smolvla --tier A --trials 20

uv run python -m grasprl.real.run --method ppo \
    --checkpoint runs/ppo_seed0/checkpoints/last/pretrained_model --tier A --trials 20

uv run python -m grasprl.real.run --method gaf \
    --checkpoint checkpoints/base_smolvla --critic runs/gaf_critic --tier A --trials 20

uv run python -m grasprl.real.metrics runs/real_ppo/eval/results.csv
```

The base policy is **re-anchored inside this run** rather than compared against the historical 53%,
because all three arms run at `n_exec = 10` while the original number was measured with the
checkpoint's native 50-step chunk.

---

## How each method works

### Arm A: Flow-SDE PPO -- `grasprl/ppo/`

SmolVLA generates a chunk by integrating a deterministic flow ODE, so there is no action density
and nothing for PPO to clip. Replacing it with the marginal-preserving reverse-time SDE gives a
chain of Gaussian transitions with an exact log-density that still samples the distribution
imitation learned:

```
dx = [ v(x,t) - (g(t)^2 / 2) * score(x,t) ] dt + g(t) dW      score(x,t) = -((1-t)v + x)/t
```

One env step is one policy decision (`n_exec` actions executed); reward, GAE and the value head
live there, while the PPO ratio is formed **per inner denoise step** for K times finer clipping.
The action expert and its four projections train; SigLIP and SmolLM2 stay frozen.

Reward is potential-based, with `drop_penalty` raised to 15 so that a lifted-then-dropped cube is
worse than one never lifted -- the term that makes this run about grasp stability rather than
grasp frequency.

### Arm B: Guided Action Flow -- `grasprl/gaf/`

Nothing about the policy changes. An action-chunk critic `Q(f_o, a_{0:H-1}, e_tau)` is regressed
onto a sparse success-to-go target on rollouts of the frozen policy, then at every denoising step
its gradient bends the velocity:

```
a_hat = x_t - t*v              # clean-action estimate            (Eq. 3)
g     = d Qbar / d a_hat       # ensemble mean                    (Eq. 6-7)
m     = max(m_min, exp(-alpha * sigma_Q))   # disagreement gate   (Eq. 8)
v_g   = v - m * clip(g, c) / beta                                 (Eq. 9)
```

Two deliberate departures from the paper, both documented in `grasprl/gaf/critic.py`:

* the critic reads the pooled VLM feature alongside the state (`features: state+pooled`) -- free,
  the paper's own stated bottleneck, and **the same feature PPO's value head consumes**, so both
  arms get value functions of equal power. `--critic-features state` reproduces the compact variant.
* the task feature is constant here (one task) and is off by default.

---

## The three traps, and the tests that pin them

Each of these produces a plausible-looking number rather than an error, which is why each has a test.

| trap | why it is silent | test |
|---|---|---|
| **The SDE drift correction subtracts.** The chain runs backwards in time; a forward-time sign anti-cancels the diffusion. | Losses look healthy. Cost a factor of 7 in the source repo. | `test_sde_preserves_the_ode_marginals`, `test_sampler_uses_the_reverse_time_sign` |
| **The guidance term subtracts too**, for the same reason: `a_hat = x - t*v`, so lowering `v` along the gradient *raises* `a_hat` along it. | Guidance silently walks downhill and just reports a worse policy. | `test_guidance_increases_the_critic_value` |
| **KL is measured per coordinate.** One decision is a 10×6 block through K steps, so summed KLs run ~100x above PPO conventions. | A stock `target_kl = 0.01` early-stops every update after ~5 of 64 minibatches. | config comments; `kl_reference` and `approx_kl` are divided by `n_scored`, the ratio never is |

Plus: the critic reads only SmolVLA's 6 physical action dims, never its 26 padding dims; the
train/val split is by **episode**, never by chunk; and guidance hyperparameters are chosen on a
validation seed range and reported once on a disjoint held-out range -- the paper's own headline
caution is a +10.0 pp validation gain that was worth +2.5 pp held out.

```bash
MUJOCO_GL=egl uv run --extra dev pytest tests/ -q      # 49 tests
```

---

## Layout

```
grasprl/
  sim/       scene.py (contact grasp + pads), calibrate.py, kinematics, ik, expert, randomization
  envs/      pickplace_env.py, vec_env.py, rules.py (failure taxonomy), reward.py
  policy/    smolvla_flow_sde.py (the SDE sampler), actor.py (one interface per arm), heads, loader
  ppo/       train_ppo.py                      Arm A
  gaf/       collect, critic, train_critic, guided_sampler, sweep      Arm B
  eval/      evaluate.py, report.py, plots.py
  real/      run.py, harness.py, infer.py, metrics.py   operator-scored, same CSV schema as sim2real
configs/     scene.yaml  randomization.yaml  ppo_flow_sde.yaml  gaf.yaml  eval_real.yaml
```

Everything is vendored rather than imported across repos, so this one stands alone.

## Reading the result honestly

* The sim comparison is a comparison **under the calibrated contact model**, with acquisition
  abstracted. It is where the statistics are; it is not the ground truth.
* 20 real trials per arm is noisy: a difference under about 15 pp is not significant.
* Success rate alone cannot answer "did the grasp get better" -- a method that grasps more often
  but holds no better, and one that holds better but attempts fewer grasps, post the same number.
  That is why every table reports `lift rate`, `grasp_slip` and `grabbed_nothing` beside it.
* A null result for either method is a result. The PPO machinery here produced a **negative**
  result in the repo it came from (35.2% → 31.2%, statistically identical); the difference now is
  that the reward penalises the exact failure being targeted and the sim can finally express it.
