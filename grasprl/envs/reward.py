"""Potential-based reward shaping for the pick-and-place task.

Shaping is potential-based (Ng, Harada & Russell 1999): every dense term enters
only as ``gamma * Phi(s') - Phi(s)``, which provably leaves the optimal policy
unchanged. An earlier unpublished project of ours learned this the
expensive way -- with a raw progress bonus, episode return climbed while success
*halved*, because the policy farmed the shaping instead of finishing the task.

Event terms (slip, success, time) sit outside the potential, so they do change
the objective. That is deliberate: ``drop_penalty`` tells PPO that a
lifted-then-dropped cube is worse than never lifting one.

It has to stay **small**, though, and that is not obvious. Because the shaping is
potential-based it telescopes over an episode, so a grasp that is carried most of
the way and then dropped earns essentially no shaping to offset the penalty --
the penalty is the entire return. Size it against ``success_bonus`` and the
*exploration-time* retention rate, not the evaluation-time one:

    EV(attempt) = retention * success_bonus - (1 - retention) * drop_penalty

If that is negative at the retention the policy actually achieves while
exploring, the optimal policy is to stop grasping, and PPO will find it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from grasprl.envs import rules


@dataclass
class RewardConfig:
    """Weights for the shaped reward.

    ``gamma`` is intentionally NOT settable from YAML -- it must equal the PPO
    discount or the shaping stops being policy-invariant, so
    :func:`grasprl.ppo.train_ppo.load_run_config` copies it from ``ppo.gamma``
    and raises if a config file tries to set it here.
    """

    gamma: float = 0.99
    progress_weight: float = 3.0
    distance_scale: float = 0.10   # m; e-folding distance of the reach/carry terms
    success_bonus: float = 100.0
    # Small on purpose. Shaping is potential-based and telescopes to ~0 over a
    # grasp-carry-drop episode, so this IS the whole return of a failed grasp.
    # At 15.0 it drove the lift rate to zero: with exploration-time retention at
    # 0.00, attempting a grasp was worth -15 against 0 for never trying. See
    # configs/ppo_flow_sde.yaml for the arithmetic.
    drop_penalty: float = 2.0      # a slip, the failure this repo targets
    time_penalty: float = 0.01     # per control tick
    knock_penalty: float = 5.0

# Fractions of the potential earned by each phase. Reaching the cube is worth
# less than carrying it, and carrying is worth less than seating it, so the
# potential is monotone along a competent trajectory.
REACH_FRACTION = 0.4
CARRY_FRACTION = 0.9


def potential(scene, state: rules.EpisodeState, cfg: RewardConfig) -> float:
    """Phi(s): how far along the task the current state is, in [0, 1] * weight."""
    if state.success:
        frac = 1.0
    elif state.held is not None:
        # Carrying: close the gap between the held cube and the cup.
        cube = scene.body_xpos(f"cube_{state.held}")
        cup = scene.body_xpos("cup")
        d = float(math.dist(cube[:2], cup[:2]))
        frac = REACH_FRACTION + (CARRY_FRACTION - REACH_FRACTION) * math.exp(-d / cfg.distance_scale)
    else:
        # Reaching: close the gap between the grasp point and the nearer cube.
        _, d = rules.nearest_cube(scene)
        frac = REACH_FRACTION * math.exp(-d / cfg.distance_scale)
    return cfg.progress_weight * frac


def step_reward(prev_phi: float, next_phi: float, state: rules.EpisodeState,
                before: Snapshot, ticks: int, cfg: RewardConfig) -> float:
    """Reward for one decision (``ticks`` control ticks of one action chunk).

    Event terms are charged on the *delta* against ``before``, a snapshot taken
    at the start of the decision. Charging them on the current flags instead
    would bill the cup-knock penalty on every remaining decision of the episode
    and would miss a success that a later tick in the same chunk undid.
    """
    r = cfg.gamma * next_phi - prev_phi
    r -= cfg.time_penalty * ticks
    r -= cfg.drop_penalty * (state.slips - before.slips)
    if state.success and not before.success:
        r += cfg.success_bonus
    if state.cup_knocked and not before.cup_knocked:
        r -= cfg.knock_penalty
    return float(r)


@dataclass
class Snapshot:
    """The event counters at the start of a decision, for delta accounting."""

    slips: int
    success: bool
    cup_knocked: bool

    @classmethod
    def of(cls, state: rules.EpisodeState) -> Snapshot:
        return cls(slips=state.slips, success=state.success, cup_knocked=state.cup_knocked)
