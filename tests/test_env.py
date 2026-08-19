"""The environment contract, and the failure classifier the whole comparison reads.

The classifier is the measuring instrument here: "did the grasp get better" is
answered by the split between ``grasp_slip`` and ``grabbed_nothing``, so a
mislabelled episode is a corrupted result rather than a cosmetic bug. These
tests drive it with hand-built tick sequences through a stub scene, so they pin
the labelling rules independently of whether MuJoCo happens to grasp anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from grasprl.envs import rules
from grasprl.envs.reward import RewardConfig, Snapshot, potential, step_reward


class _StubScene:
    """Minimum surface :func:`rules.update` touches, driven from a script."""

    def __init__(self):
        self.cfg = {"cup": {"inner_radius": 0.035, "height": 0.070},
                    "cubes": {"size": 0.015},
                    "table": {"center_x": 0.28, "half_x": 0.30, "half_y": 0.32}}
        self.held = None
        self.heights = {"left": 0.015, "right": 0.015}
        self.positions = {"left": np.array([0.2, 0.1, 0.015]),
                          "right": np.array([0.2, -0.1, 0.015]),
                          "cup": np.array([0.33, 0.0, 0.0])}
        self.in_cup = {"left": False, "right": False}
        self.tilt = 0.0

    def gripped_cube(self):
        return self.held

    def cube_height(self, cube):
        return self.heights[cube]

    def body_xpos(self, name):
        return self.positions[name.removeprefix("cube_")]

    def cube_in_cup(self, cube):
        return self.in_cup[cube]

    def cup_tilt(self):
        return self.tilt

    def grasp_xpos(self):
        return np.array([0.22, 0.0, 0.05])

    def grasp_offset(self, cube):
        return 0.003

    def grasp_yaw_err(self, cube):
        return 0.1


def _fresh():
    return _StubScene(), rules.EpisodeState()


def test_never_closing_is_froze_no_attempt():
    sc, st = _fresh()
    for _ in range(10):
        rules.update(st, sc, action_grip=80.0)
    assert rules.classify(st) == rules.FROZE_NO_ATTEMPT


def test_closing_on_empty_space_is_grabbed_nothing():
    sc, st = _fresh()
    for _ in range(10):
        rules.update(st, sc, action_grip=19.0)
    assert rules.classify(st) == rules.GRABBED_NOTHING


def test_gripping_without_lifting_is_grabbed_nothing():
    """Touching the cube but never picking it up is behaviourally the same miss.

    Counting it as its own category would split the baseline's dominant failure
    across two rows and make the comparison harder to read, not easier.
    """
    sc, st = _fresh()
    sc.held = "right"
    for _ in range(10):
        rules.update(st, sc, action_grip=19.0)
    assert st.ever_gripped and not st.ever_lifted
    assert rules.classify(st) == rules.GRABBED_NOTHING


def test_lifting_then_dropping_away_from_the_cup_is_grasp_slip():
    sc, st = _fresh()
    sc.held = "right"
    sc.heights["right"] = 0.12
    sc.positions["right"] = np.array([0.25, -0.05, 0.12])   # nowhere near the cup
    for _ in range(3):
        rules.update(st, sc, action_grip=19.0)
    sc.held = None                                          # lost it
    rules.update(st, sc, action_grip=19.0)
    assert st.slips == 1
    assert rules.classify(st) == rules.GRASP_SLIP


def test_releasing_over_the_cup_is_not_a_slip():
    """The bug this guards: at the instant the jaws open the cube is still in the
    air, so testing ``cube_in_cup`` there would score every success as a slip."""
    sc, st = _fresh()
    sc.held = "right"
    sc.heights["right"] = 0.12
    for _ in range(3):
        rules.update(st, sc, action_grip=19.0)
    sc.positions["right"] = np.array([0.33, 0.0, 0.05])     # directly over the cup
    sc.held = None
    rules.update(st, sc, action_grip=60.0)
    assert st.slips == 0
    sc.in_cup["right"] = True
    rules.update(st, sc, action_grip=60.0)
    assert rules.classify(st) == rules.SUCCESS


def test_carrying_but_never_releasing_is_missed_cup():
    sc, st = _fresh()
    sc.held = "right"
    sc.heights["right"] = 0.12
    for _ in range(10):
        rules.update(st, sc, action_grip=19.0)
    assert rules.classify(st) == rules.MISSED_CUP


def test_action_bounds_cover_the_calibrated_range():
    from grasprl.sim import kinematics as K

    lo, hi = rules.action_bounds(K)
    assert lo.shape == hi.shape == (6,)
    assert (lo[:5] == [K.REACHABLE_DEG[j][0] for j in K.ARM_JOINTS]).all()
    assert lo[5] == 0.0 and hi[5] == 100.0


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def test_event_terms_are_charged_once_per_decision():
    """Charging on the flags rather than the delta would bill the cup-knock
    penalty on every remaining decision of the episode."""
    _sc, st = _fresh()
    cfg = RewardConfig()
    st.cup_knocked = True
    st.success = True
    before = Snapshot(slips=0, success=True, cup_knocked=True)
    r = step_reward(0.0, 0.0, st, before, ticks=10, cfg=cfg)
    assert r == pytest.approx(-cfg.time_penalty * 10)


def test_a_slip_is_penalised_and_a_success_rewarded():
    _sc, st = _fresh()
    cfg = RewardConfig()
    before = Snapshot(slips=0, success=False, cup_knocked=False)
    st.slips = 1
    slip_r = step_reward(0.0, 0.0, st, before, ticks=0, cfg=cfg)
    assert slip_r == pytest.approx(-cfg.drop_penalty)

    st.slips = 0
    st.success = True
    win_r = step_reward(0.0, 0.0, st, before, ticks=0, cfg=cfg)
    assert win_r == pytest.approx(cfg.success_bonus)
    assert win_r > -slip_r, "success must outweigh the shaping penalties"


def test_potential_is_monotone_along_a_competent_trajectory():
    """Reaching < carrying < seated, so the shaping never rewards going backwards."""
    sc, st = _fresh()
    cfg = RewardConfig()
    far = potential(sc, st, cfg)

    sc.positions["right"] = np.array([0.2, -0.1, 0.015])
    st.held = None
    st.held = "right"
    carrying = potential(sc, st, cfg)
    st.success = True
    seated = potential(sc, st, cfg)
    assert far < carrying < seated
    assert seated == pytest.approx(cfg.progress_weight)


def test_opening_the_jaws_away_from_the_cup_is_missed_cup_not_slip():
    """The distinction the headline metric depends on.

    Carrying a cube to the wrong place and letting go is a planning error. Only
    losing it while the jaws are still commanded shut is a grip failure. Scoring
    the first as the second inflates the slip rate with misplacements, which
    would then be credited to whichever method improved placement.
    """
    sc, st = _fresh()
    sc.held = "right"
    sc.heights["right"] = 0.12
    for _ in range(3):
        rules.update(st, sc, action_grip=19.0)
    sc.positions["right"] = np.array([0.25, -0.05, 0.12])   # not over the cup
    sc.held = None
    rules.update(st, sc, action_grip=60.0)                  # jaws commanded OPEN
    assert st.slips == 0
    assert st.deliberate_releases == 1
    assert rules.classify(st) == rules.MISSED_CUP


def test_losing_the_cube_with_jaws_shut_is_still_a_slip():
    sc, st = _fresh()
    sc.held = "right"
    sc.heights["right"] = 0.12
    for _ in range(3):
        rules.update(st, sc, action_grip=19.0)
    sc.positions["right"] = np.array([0.25, -0.05, 0.12])
    sc.held = None
    rules.update(st, sc, action_grip=19.0)                  # jaws still SHUT
    assert st.slips == 1 and st.deliberate_releases == 0
    assert rules.classify(st) == rules.GRASP_SLIP
