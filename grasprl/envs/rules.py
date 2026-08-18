"""Episode bookkeeping: what happened, and what to call it.

The whole point of this repo is that ``grasp_slip`` is a *distinct* failure from
``grabbed_nothing`` -- the real-arm evaluation of the base policy splits 30% slip
vs 13% grabbed-nothing, and a method that fixes one but not the other should be
visible as such. So the sim classifier reproduces the real harness's categories
by name (:class:`grasprl.real.harness.FailureCategory`), which is what makes the
sim and real tables in ``results/comparison.md`` directly comparable.

The classification is *first-cause*, matching the operator's instruction on the
real rig: the earliest thing that went wrong wins, not the last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Category names, kept byte-identical to the real harness's enum values.
SUCCESS = "success"
GRABBED_NOTHING = "grabbed_nothing"
GRASP_SLIP = "grasp_slip"
MISSED_CUP = "missed_cup"
KNOCKED_CUP_OVER = "knocked_cup_over"
COLLISION_UNSAFE = "collision_unsafe"
FROZE_NO_ATTEMPT = "froze_no_attempt"
TIMEOUT = "timeout"

CATEGORIES = (
    GRABBED_NOTHING, GRASP_SLIP, MISSED_CUP,
    KNOCKED_CUP_OVER, COLLISION_UNSAFE, FROZE_NO_ATTEMPT, TIMEOUT,
)


@dataclass
class EpisodeState:
    """Running per-episode facts the reward and the classifier both read.

    Lives for one episode and is updated once per control tick by
    :func:`update`. Keeping it separate from the env means the classifier can be
    unit-tested against a hand-built tick sequence with no MuJoCo at all.
    """

    lift_height: float = 0.05        # m above the table that counts as "lifted"
    grip_closed_below: float = 30.0  # RANGE_0_100 command that counts as "closing"
    cup_tilt_limit: float = 0.35     # rad (~20 deg) before the cup counts as knocked over

    # -- accumulated facts --------------------------------------------------
    ticks: int = 0
    ever_closed: bool = False        # did the policy ever command the gripper shut
    ever_gripped: bool = False       # did both pads ever hold a cube
    ever_lifted: bool = False        # did a gripped cube ever clear lift_height
    held: str | None = None          # cube currently gripped
    lifted_side: str | None = None   # cube that was lifted (first one wins)
    slips: int = 0                   # gripped+lifted -> released outside the cup
    cup_knocked: bool = False
    success: bool = False
    newly_successful: bool = False   # set for exactly one tick, for the reward
    # Grasp quality at the moment the cube was first gripped, for diagnostics.
    grasp_offset_at_pickup: float = float("nan")
    grasp_yaw_err_at_pickup: float = float("nan")
    events: list[str] = field(default_factory=list)


def update(state: EpisodeState, scene, action_grip: float) -> EpisodeState:
    """Advance ``state`` by one control tick. Call after ``scene.step``."""
    state.ticks += 1
    state.newly_successful = False

    if action_grip < state.grip_closed_below:
        state.ever_closed = True

    held = scene.gripped_cube()
    if held is not None:
        if not state.ever_gripped:
            state.grasp_offset_at_pickup = scene.grasp_offset(held)
            state.grasp_yaw_err_at_pickup = scene.grasp_yaw_err(held)
        state.ever_gripped = True
        if scene.cube_height(held) > state.lift_height:
            if not state.ever_lifted:
                state.lifted_side = held
                state.events.append(f"lifted:{held}@{state.ticks}")
            state.ever_lifted = True
    elif state.held is not None and state.ever_lifted:
        # Was holding a lifted cube, now holding nothing: either a deliberate
        # release over the cup or a slip. `cube_in_cup` cannot decide this yet --
        # at the instant the jaws open the cube is still in the air, so testing
        # it here would score every successful placement as a slip. Judge by
        # *where* the release happened instead.
        if not _released_over_cup(scene, state.held):
            state.slips += 1
            state.events.append(f"slip:{state.held}@{state.ticks}")
    state.held = held

    if scene.cup_tilt() > state.cup_tilt_limit:
        state.cup_knocked = True

    if not state.success:
        for side in ("left", "right"):
            if scene.cube_in_cup(side):
                state.success = True
                state.newly_successful = True
                state.events.append(f"success:{side}@{state.ticks}")
                break
    return state


def _released_over_cup(scene, cube: str) -> bool:
    """True if ``cube`` was let go in a position from which it falls into the cup.

    Horizontal containment plus "not below the rim": a cube released off to the
    side, or one that has already dropped past the rim on its way to the table,
    was not placed.
    """
    c = scene.body_xpos(f"cube_{cube}")
    cup = scene.body_xpos("cup")
    cfg = scene.cfg["cup"]
    radial = float(np.hypot(c[0] - cup[0], c[1] - cup[1]))
    return radial < cfg["inner_radius"] and (c[2] - cup[2]) > 0.0


def classify(state: EpisodeState) -> str:
    """First-cause label for a finished episode.

    Order matters and mirrors the operator's rule on the real rig:

    1. it worked;
    2. it never even tried;
    3. it closed on empty space (the co-trained policy's other failure);
    4. it had the cube up and dropped it -- the failure this repo targets;
    5. it carried the cube but put it in the wrong place;
    6. it tipped the cup;
    7. nothing identifiable happened before the clock ran out.

    ``knocked_cup_over`` is checked *after* ``grasp_slip`` on purpose: if the arm
    dropped the cube and then blundered into the cup, the drop is the first
    cause.
    """
    if state.success:
        return SUCCESS
    if not state.ever_closed:
        return FROZE_NO_ATTEMPT
    if not state.ever_gripped:
        return GRABBED_NOTHING
    if state.slips > 0:
        return GRASP_SLIP
    if state.ever_lifted:
        return MISSED_CUP
    if state.cup_knocked:
        return KNOCKED_CUP_OVER
    # Gripped but never lifted: the jaws touched the cube and it never left the
    # table. Behaviourally that is the same miss as closing on empty space.
    return GRABBED_NOTHING


def action_bounds(kinematics) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint (low, high) for a policy action in LeRobot calibrated units.

    Joints 1-5 come from the real calibrated range of motion; the gripper is
    RANGE_0_100. Clipping to this is what stops a diverging policy from
    commanding a pose the real servos would refuse.
    """
    lo, hi = [], []
    for j in kinematics.ARM_JOINTS:
        a, b = kinematics.REACHABLE_DEG[j]
        lo.append(a)
        hi.append(b)
    lo.append(0.0)
    hi.append(100.0)
    return np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32)


def out_of_workspace(scene) -> bool:
    """True if any cube has left the table -- an unrecoverable episode."""
    t = scene.cfg["table"]
    x_lim = (t["center_x"] - t["half_x"], t["center_x"] + t["half_x"])
    y_lim = (-t["half_y"], t["half_y"])
    for side in ("left", "right"):
        p = scene.body_xpos(f"cube_{side}")
        if not (x_lim[0] < p[0] < x_lim[1] and y_lim[0] < p[1] < y_lim[1]) or p[2] < -0.02:
            return True
    return False


def nearest_cube(scene) -> tuple[str, float]:
    """``(side, distance)`` from the grasp point to the closer cube (m)."""
    p = scene.grasp_xpos()
    best, best_d = "left", math.inf
    for side in ("left", "right"):
        d = float(np.linalg.norm(p - scene.body_xpos(f"cube_{side}")))
        if d < best_d:
            best, best_d = side, d
    return best, best_d
