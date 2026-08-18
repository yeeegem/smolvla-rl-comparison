"""The MuJoCo-to-LeRobot units bridge.

Every action the policy emits and every state it reads passes through here, so a
sign flip or a scale error would silently corrupt both the demonstrations and the
RL rollouts while everything still ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from grasprl.sim import kinematics as K


def test_round_trip_is_identity():
    rng = np.random.default_rng(0)
    for _ in range(100):
        qpos = np.concatenate([
            rng.uniform(-1.5, 1.5, 5),
            rng.uniform(K.GRIPPER_LO_RAD, K.GRIPPER_HI_RAD, 1),
        ])
        back = K.state_to_ctrl(K.qpos_to_state(qpos))
        assert np.allclose(back, qpos, atol=1e-6)


def test_arm_joints_are_plain_degrees():
    """Joints 1-5 use LeRobot's DEGREES mode with no offset, so the conversion is
    exactly radians to degrees."""
    qpos = np.array([0.5, -0.5, 1.0, -1.0, 0.25, 0.0])
    state = K.qpos_to_state(qpos)
    assert np.allclose(state[:5], np.rad2deg(qpos[:5]), atol=1e-5)


def test_gripper_maps_onto_range_0_100():
    assert K.qpos_to_state(np.zeros(6))[5] == pytest.approx(
        100 * (0 - K.GRIPPER_LO_RAD) / (K.GRIPPER_HI_RAD - K.GRIPPER_LO_RAD), abs=1e-4)
    lo = K.qpos_to_state(np.array([0, 0, 0, 0, 0, K.GRIPPER_LO_RAD]))[5]
    hi = K.qpos_to_state(np.array([0, 0, 0, 0, 0, K.GRIPPER_HI_RAD]))[5]
    assert lo == pytest.approx(0.0, abs=1e-4)
    assert hi == pytest.approx(100.0, abs=1e-4)


def test_gripper_range_saturates():
    """Out-of-range hinge angles clamp instead of extrapolating, so a policy that
    over-commands the gripper cannot drive the value outside 0..100."""
    below = K.qpos_to_state(np.array([0, 0, 0, 0, 0, K.GRIPPER_LO_RAD - 1.0]))[5]
    above = K.qpos_to_state(np.array([0, 0, 0, 0, 0, K.GRIPPER_HI_RAD + 1.0]))[5]
    assert below == pytest.approx(0.0)
    assert above == pytest.approx(100.0)


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError):
        K.qpos_to_state(np.zeros(5))
    with pytest.raises(ValueError):
        K.state_to_ctrl(np.zeros(7))


def test_degree_range_matches_the_lerobot_formula():
    lo, hi = K.lerobot_degree_range(1000, 3000)
    assert lo == pytest.approx(-1000 * 360 / 4095)
    assert hi == pytest.approx(1000 * 360 / 4095)
