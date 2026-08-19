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

**The deliverable is the sim comparison.** Both methods start from the same frozen checkpoint,
train and are scored in the same simulator, against the same sim baseline, with the same failure
taxonomy. Real-arm numbers appear throughout as **reference only**: they are why grasp slip is the
failure being targeted, and the repo carries a working real-arm harness if you want to spot-check
a result, but the comparison is decided in sim and the sim is not required to reproduce them.

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

This is a fact about the sim, not a defect to be fixed before proceeding. It sets the acquisition
rate, and therefore how many episodes the comparison needs; it does not stop the sim being a fair
testbed, because both methods face exactly the same handicap.

So the two failure modes are separated, and only one of them is abstracted:

* **Acquisition** -- did the cube end up between the jaws. Abstracted. `Scene.capture()` cancels
  the *across-the-jaws* component of the placement error when the gripper closes within
  `grasp.capture_radius`, set to leave enough episodes reaching the retention stage to measure it.
* **Retention** -- did it stay there. **Real physics**, untouched. Grasp height, cube yaw,
  gripper command and carry acceleration all still decide the outcome, and those are exactly the
  variables a policy can learn to fix.

`Scene.capture` corrects one axis and one axis only; it does not weld, and it does not tidy up the
grasp. Close high on the cube and the jaws still catch its top edge. Meet a corner instead of a
face and it still squirts out. Command the gripper two units too open and it still drops.

Set `grasp.capture_radius: null` for a fully unassisted grasp, or `grasp.mode: weld` for the
original abstraction, and re-run any result you want to check against a different substrate.

---

## Status: baseline chosen, both methods ready to run

The 12-cell sweep is done (40 episodes per cell) and the baseline is frozen in
`configs/scene.yaml`: `capture_radius: 0.02`, `gripper_forcerange: 1.4`, `pad_friction: 0.6`.

The sim does not have to reproduce the real arm; it has to be a testbed where a method can show
an effect and where that effect can be measured. `grasprl.sim.calibrate` scores cells on exactly
that, as `headroom x sample_yield`:

* **headroom** peaks when baseline retention is mid-range. Pinned near 0 or 1 there is nowhere for
  a method to move, and a proportion is least sensitive at its extremes.
* **sample yield** is the acquisition rate, because retention is only observed on episodes that
  get a cube off the table. It sets how many usable samples an episode budget buys.

The chosen cell is the best on that criterion, and it is also the cell closest to the real arm,
so nothing was traded away:

| | acquisition | retention |
|---|---|---|
| **chosen sim baseline** | **0.34 to 0.45** | **0.37 to 0.51** |
| real arm (reference) | 0.83 | 0.64 |
| weld ceiling (reference) | n/a | grasp cannot fail |

The two stages behave very differently, which is worth knowing before reading any result:

* **Retention responds cleanly** to `gripper_forcerange`, monotonically, across two independent
  rows of the grid. It is the grasp-slip question and the headline metric.
* **Acquisition is close to inert.** Every `capture_radius` from 20 mm to 45 mm gives 0.30 to
  0.50, all within noise at 40 episodes. The limit is that the policy often never commands the
  gripper shut near a cube on sim images, which is a near-miss no tolerance knob can catch. See
  the section above for the measurement.

**The consequence that matters: statistical power.** Retention is measured on roughly a third of
episodes, so a 200-episode run yields only about 68 retention samples.

| effect | retention samples/arm | **total episodes/arm** |
|---|---|---|
| +10 pp retention | 382 | ~1120 |
| +15 pp | 172 | ~510 |
| +20 pp | 97 | ~285 |

At 200 episodes an observed +15 pp swing is **not** significant (p = 0.08). Every rate in
`results/comparison.md` therefore carries a 95% Wilson interval, and each method is compared to
the baseline with a two-proportion test. Read the interval, not the point estimate.

**The live risk.** With roughly two thirds of episodes ending in `grabbed_nothing`, PPO's reward
is dominated by "attempt a grasp at all" rather than "hold on to it". A method could win on
acquisition rather than retention. The evaluator and the report lead with retention and show
acquisition beside it, so a win of the wrong kind is visible rather than buried in a success rate.

**One correction found while watching rollouts.** The classifier was scoring any release away from
the cup as `grasp_slip`, including deliberate releases a centimetre short of the rim. Those are
`missed_cup`, a planning error rather than a grip failure, and counting them as slip would have
inflated the headline metric and credited it to whichever method improved placement.
`rules.update` now separates them by the gripper command: jaws opening means a deliberate release,
jaws still shut means a genuine slip. Both cases are pinned by tests.

| | state |
|---|---|
| contact grasp, pads, masks, weld fallback | built, tested, verified against the MJCF |
| env, failure taxonomy, potential-based reward | built, tested |
| baseline selection | **done**, frozen in `configs/scene.yaml` |
| Arm A: Flow-SDE PPO | built; ~70 ticks/s, a 400k-tick run is ~1.6 h |
| Arm B: Guided Action Flow | built; collect to critic to guided sampling verified end to end |
| sim evaluator with Wilson intervals, rollout viewer, report | built |
| real-arm harness (reference, optional) | built |
| the two RL runs | **not started** |

## Quickstart

```bash
uv sync --extra dev
scripts/fetch_base_checkpoint.sh              # copies the frozen SmolVLA checkpoint

MUJOCO_GL=egl uv run --extra dev pytest tests/ -q
MUJOCO_GL=egl uv run python scripts/tune_jaw_pads.py --out scene_views/grasp.png
```

### 1. Choose the sim baseline -- already done, re-run only to change it

The chosen cell is frozen in `configs/scene.yaml`. Both methods are scored in this simulator, so
re-running this after seeing a result would invalidate the comparison.

```bash
bash scripts/calibrate_slip.sh                # ~40 min; rewrites configs/scene.yaml
```

Sweeps `capture_radius` against `gripper_forcerange` and scores each cell on **fitness as a
testbed**, `headroom x sample_yield`: baseline retention away from the floor and ceiling so a
method has room to move, and an acquisition rate high enough that enough episodes reach the
retention stage to measure it. It refuses to write a cell where retention is pinned at an extreme
or almost nothing is ever lifted, because neither method could show an effect there. The real-arm
profile is printed alongside as reference, not as a target.

### 1b. Look at the baseline before spending GPU on it

```bash
bash scripts/watch.sh base                    # -> results/rollout_base.mp4
EPISODES=8 SEED=42 bash scripts/watch.sh base
```

Every frame carries the commanded gripper value and the jaw gap it produces, the distance from
the grasp point to the nearer cube against the capture radius, whether both pads are loaded and
with how much force, and the cube height. That is enough to tell a near-miss from a mistimed
close from a genuine slip without guessing. The same command works for `ppo` and `gaf` once
those exist, so the three can be watched side by side.

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

### 5. Optional: spot-check on the real SO-ARM101

**Not part of the deliverable.** The comparison is decided by step 4. This exists if you want to
see whether a sim result survives contact with the hardware; 20 trials cannot settle a difference
between two methods (a gap under ~15 pp is not significant at that size), but it can tell you
whether a method has broken something obvious.

Edit `configs/eval_real.yaml` for your port and cameras. All three arms run through one control
loop at the same decision cadence as the sim.

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

* **The comparison is the sim comparison.** Both methods start from the same frozen checkpoint,
  train and are scored in the same sim against the same baseline. Real-arm numbers are reference
  context for why grasp slip is the target, not a bar the sim has to clear.
* **Report retention, not success rate.** Success mixes the two stages of the task, and is
  dominated by acquisition, which barely responds to anything. Retention is the grasp-slip
  question.
* **Read the interval, not the point estimate.** Retention is measured on about a third of
  episodes. At 200 episodes per arm its 95% interval is roughly +/- 11 points, so only a very
  large effect is visible. Size the run against the power table above before believing a gap.
* **A method can win for the wrong reason.** There is far more headroom in acquisition than in
  retention. If an arm improves success mostly by attempting more grasps while retention stays
  flat, say so: that is not evidence about grasp stability.
* **A null result for either method is a result.** The PPO machinery here produced a negative
  result in the repo it came from (35.2% to 31.2%, statistically identical). The difference now is
  that the reward penalises the exact failure being targeted and the sim can express it, but that
  is a reason to run the experiment, not to expect a particular answer.
