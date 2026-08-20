# Flow-SDE PPO vs Guided Action Flow, on SmolVLA grasp stability

Both arms start from the **same frozen checkpoint** (`checkpoints/base_smolvla`), which scores 53% on the real SO-ARM101 and loses 30% of trials to grasp slip. They differ in where the improvement is allowed to live: PPO updates 99.8M policy parameters; Guided Action Flow updates none and steers the sampler with a learned action-chunk critic instead.

## Conclusion

* **Flow-SDE PPO: no measurable improvement in retention.** 30.6% -> 26.7%, a change of -4.0% [-13.4%, +5.5%], p = 0.41. The interval spans zero, so this is a null result rather than a demonstrated regression.
* **Guided Action Flow: no measurable improvement in retention.** 30.6% -> 21.0%, a change of -9.6% [-19.2%, -0.0%], p = 0.05. The interval spans zero, so this is a null result rather than a demonstrated regression.

* **PPO did change the policy, but not in the way that was asked for.** Acquisition rose 32.0% -> 39.0% (+7.0% [+1.1%, +12.9%], p = 0.021), which is significant, while retention did not move. It learned to attempt more grasps, not to hold on to them. Acquisition is the half of this sim that is a known artifact of the policy's placement error, so improving it is not evidence about grasp stability, and net success rate barely changed as a result.

**Neither method fixed grasp slip.** Success rate is statistically indistinguishable from the frozen baseline for PPO (9.8% -> 10.4%, p = 0.75) and no better for Guided Action Flow (9.8% -> 6.6%, p = 0.07).

### What would have to be true to overturn this

* **Power.** At 500 episodes per arm this comparison could only resolve a retention gain of roughly 20 points. A real but modest improvement of 5 to 10 points would not have been visible. The table below gives the episode counts that would be needed.
* **PPO's budget.** 500k control ticks is small for on-policy RL on a 450M policy. The training curve oscillated rather than converged, and the checkpoint was chosen from a noisy 20-episode window; the 500-episode evaluation says that peak did not hold.
* **The critic's data.** Guided Action Flow's critic saw 600 frozen-policy episodes, of which only about a third reached the retention stage at all. Its final validation MSE and near-zero ensemble disagreement suggest it fit the sparse success-to-go target without learning to rank near-miss grasps, which is the paper's own stated bottleneck.


## Sim (contact-physics grasp, held-out seeds)

| arm | **retention** (95% CI) | vs baseline | acquisition | success | grasp_slip | grabbed_nothing |
|---|---|---|---|---|---|---|
| frozen SmolVLA (baseline) | **30.6%** [24%, 38%] (49/160) | baseline | 32.0% | 9.8% | 21.2% | 68.0% |
| + Flow-SDE PPO (weights updated) | **26.7%** [21%, 33%] (52/195) | -4.0% [-13%, +6%], p=0.41, not significant | 39.0% | 10.4% | 26.8% | 61.0% |
| + Guided Action Flow (weights frozen) | **21.0%** [15%, 28%] (33/157) | -9.6% [-19%, -0%], p=0.05, not significant | 31.4% | 6.6% | 23.0% | 68.6% |

Real arm, for reference only: retention 64% (16/25), acquisition 83% (25/30). The sim is not required to match it; both methods face the same sim.

**Power.** Episodes per arm needed to resolve a retention gain, at the baseline's 31% and an acquisition rate of 32%:

| effect | lifted episodes per arm | total episodes per arm |
|---|---|---|
| +10% retention | 359 | ~1122 |
| +15% retention | 164 | ~512 |
| +20% retention | 94 | ~294 |

Read the interval, not the point estimate. A method whose interval overlaps the baseline's has not been shown to help.

## Real SO-ARM101 (reference only, operator-scored, tier A)

Not the deliverable. The comparison is decided by the sim table above; 20 trials per arm cannot separate two methods. This is a spot-check that a sim result has not broken something obvious on hardware.

_No real-arm results yet. Run `grasprl.real.run` for each arm._

## The simulator these numbers come from

The sim's job is to be a fair, discriminative testbed, not to reproduce the real arm. Both methods face identical physics, so the comparison is valid within it. Two properties are worth stating because they bound what the result means.

**The task has two stages and they are not equally trustworthy.**

| | sim baseline | real arm (reference) |
|---|---|---|
| acquisition (got a cube off the table) | 45% | 83% |
| **retention** (of those, reached the cup) | **50%** | **64%** |

Retention is the grasp-slip question and the metric this comparison reports. Acquisition is roughly half the real arm's, because the frozen policy places its jaws about ten times less accurately in MuJoCo than on hardware; pad size, friction, servo torque and contact softness were all swept and none of them move it. That gap is why a method can raise acquisition without that meaning anything about grasp stability.

Chosen by fitness as a testbed (headroom x sample yield): `capture_radius = 0.02`, `gripper_forcerange = 1.4`, `pad_friction[0] = 0.6`, then frozen for both arms.

**Slip is only representable at all because of the contact model.** The simulator this checkpoint was trained in welds the cube to the gripper, so a grasp cannot fail there. This repo replaces that with two high-friction jaw pads masked to collide only with cubes, anchored to measured MJCF geometry: the jaw gap is 29.6 mm at gripper command 19 against a 30 mm cube, and 32.4 mm at 21. Hold-versus-drop turns on about two units of one action dimension.

## Arm A: Flow-SDE PPO

- 391 updates, 500,480 control ticks, 1734 episodes, 3.7 h
- final training success 0.0%, slip 20.0%
- Evaluated with the deterministic ODE sampler, not the SDE it trained with: the exploration noise has a real cost and scoring it would measure the exploration mechanism, not the learned policy.

## Arm B: Guided Action Flow

- critic: 4 x 768 MLP, ensemble K=3, 30 epochs, features `state+pooled`
- trained on 600 frozen-policy episodes (17,445 chunks), episode-level split
- final val MSE 0.01663, ensemble disagreement 0.0131
- Guidance hyperparameters were chosen on the **validation** seed range and reported once on the disjoint held-out range.
