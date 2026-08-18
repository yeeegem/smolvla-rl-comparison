"""Units bridge: MuJoCo joint space <-> LeRobot calibrated state/action.

This is the make-or-break layer for zero-shot transfer. A policy trained on
simulated demonstrations is deployed unchanged on the real SO-ARM101, so the
numbers it sees and emits in sim must equal the numbers the real arm produces /
consumes for the *same physical configuration*.

LeRobot's calibrated convention (see ``lerobot.motors.motors_bus._normalize`` and
``lerobot.motors.feetech.feetech``):

- Joints 1-5 use ``MotorNormMode.DEGREES``:
      degrees = (Present_Position - mid) * 360 / (resolution - 1)
  where ``Present_Position = Actual_Position - Homing_Offset`` (the firmware
  subtracts the homing offset), ``mid = (range_min + range_max) / 2``, and
  ``resolution = 4096`` for the STS3215. The homing offset is chosen so the
  middle of the recorded range of motion sits near a half turn -> degrees are
  centred on the neutral pose.

- The gripper uses ``MotorNormMode.RANGE_0_100``:
      value = (Present_Position - range_min) / (range_max - range_min) * 100

The vendored ``so101.xml`` is the **new-calibration** model: its joint zero is the
neutral pose and its joint limits match the LeRobot degree ranges derived from a
real calibration (verified in ``tests/test_kinematics.py`` against
``valera.json`` and the dataset ``stats.json``). So for joints 1-5 the bridge is
simply ``degrees = SIGN * rad2deg(theta) + OFFSET`` with ``SIGN = +1`` and
``OFFSET = 0`` -- but both are kept as per-joint constants so a one-time hardware
sign/zero check can adjust them without touching call sites.

The gripper is a single linear map between its MJCF hinge angle and the
``RANGE_0_100`` value; the scripted expert commands open/close *in RANGE_0_100
units* (matching the real data distribution) and this module converts those to a
hinge target, so recorded gripper values transfer 1:1 to the real arm.
"""

from __future__ import annotations

import numpy as np

# Motor order: identical to the real dataset's observation.state / action and to
# the actuators in assets/so101/so101.xml.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINTS = JOINT_NAMES[:5]  # DEGREES-mode joints
GRIPPER = JOINT_NAMES[5]      # RANGE_0_100 joint

# Per-joint sign and zero-offset (degrees) for the 5 arm joints, applied as
# ``degrees = SIGN * rad2deg(theta) + OFFSET_DEG``. Defaults assume the
# new-calib MJCF zero aligns with the LeRobot calibrated neutral (verified in
# tests). A hardware check can override these if a joint reads inverted.
JOINT_SIGN: dict[str, float] = {j: 1.0 for j in ARM_JOINTS}
JOINT_OFFSET_DEG: dict[str, float] = {j: 0.0 for j in ARM_JOINTS}

# Gripper hinge actuation span (radians) in the MJCF, mapped linearly onto the
# RANGE_0_100 value. From so101.xml the gripper joint range is approx
# [-10 deg, 100 deg]; value 0 == GRIPPER_LO_RAD, value 100 == GRIPPER_HI_RAD.
GRIPPER_LO_RAD: float = np.deg2rad(-10.0)
GRIPPER_HI_RAD: float = np.deg2rad(100.0)


def qpos_to_state(qpos: np.ndarray) -> np.ndarray:
    """MuJoCo joint positions (rad) -> LeRobot state/action vector (6,).

    Joints 1-5 in calibrated degrees, gripper in RANGE_0_100. ``qpos`` must be
    the 6 actuated joint angles in ``JOINT_NAMES`` order.
    """
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if qpos.shape[0] != 6:
        raise ValueError(f"expected 6 joint angles, got {qpos.shape[0]}")
    out = np.empty(6, dtype=np.float32)
    for i, j in enumerate(ARM_JOINTS):
        out[i] = JOINT_SIGN[j] * np.rad2deg(qpos[i]) + JOINT_OFFSET_DEG[j]
    out[5] = _gripper_rad_to_range(qpos[5])
    return out


def state_to_ctrl(state: np.ndarray) -> np.ndarray:
    """LeRobot state/action vector (6,) -> MuJoCo joint targets (rad) (6,).

    Inverse of :func:`qpos_to_state`. Used to turn an expert/policy action in
    calibrated units into position-actuator setpoints.
    """
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] != 6:
        raise ValueError(f"expected 6 state values, got {state.shape[0]}")
    out = np.empty(6, dtype=np.float64)
    for i, j in enumerate(ARM_JOINTS):
        out[i] = np.deg2rad((state[i] - JOINT_OFFSET_DEG[j]) / JOINT_SIGN[j])
    out[5] = _gripper_range_to_rad(state[5])
    return out


def _gripper_rad_to_range(theta: float) -> float:
    frac = (theta - GRIPPER_LO_RAD) / (GRIPPER_HI_RAD - GRIPPER_LO_RAD)
    return float(np.clip(frac, 0.0, 1.0) * 100.0)


def _gripper_range_to_rad(value: float) -> float:
    frac = np.clip(value, 0.0, 100.0) / 100.0
    return float(GRIPPER_LO_RAD + frac * (GRIPPER_HI_RAD - GRIPPER_LO_RAD))


# --- calibration-derived reference (used by tests / validation) --------------

STS3215_RESOLUTION = 4096


def lerobot_degree_range(range_min: int, range_max: int) -> tuple[float, float]:
    """DEGREES-mode (min, max) for a joint given its calibration tick range."""
    mid = (range_min + range_max) / 2
    res = STS3215_RESOLUTION - 1
    return ((range_min - mid) * 360 / res, (range_max - mid) * 360 / res)


# Calibrated reachable range of motion (degrees) for the target arm, derived
# from the real calibration (valera.json) via lerobot_degree_range() and rounded
# outward with a small safety margin. The vendored MJCF's CAD joint limits are
# slightly *tighter* than the real range of motion (e.g. shoulder_lift reaches
# -103.2 deg in the dataset but the CAD limit is -100 deg), so scene.py widens
# the model limits to these values -- otherwise the sim could not reproduce
# every real pose the policy must imitate.
REACHABLE_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-118.0, 118.0),
    "shoulder_lift": (-106.0, 106.0),
    "elbow_flex": (-99.0, 99.0),
    "wrist_flex": (-104.0, 104.0),
    "wrist_roll": (-180.0, 180.0),
}


def apply_reachable_ranges(model) -> None:
    """Widen an SO-101 ``MjModel``'s joint + actuator limits to the calibrated
    real range of motion (:data:`REACHABLE_DEG`), in place.

    The gripper keeps its MJCF hinge range. Call once after loading the scene.
    """
    import mujoco

    for j, (lo_deg, hi_deg) in REACHABLE_DEG.items():
        lo, hi = np.deg2rad(lo_deg), np.deg2rad(hi_deg)
        ji = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        model.jnt_range[ji] = (lo, hi)
        ai = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        if ai >= 0:
            model.actuator_ctrlrange[ai] = (lo, hi)
