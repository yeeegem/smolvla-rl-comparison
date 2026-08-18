"""The contact grasp must be able to hold a cube -- and to drop one.

This repo's entire premise is that ``grasp_slip`` is a failure the sim can
express. The weld grasp it replaces could not: with the whole arm non-colliding
and the cube welded to the gripper, no grasp ever fails. So these tests check
both directions -- a good grasp holds, a bad one does not -- because a contact
model that always holds is just a slower weld, and one that never holds makes
every downstream number zero.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from grasprl.sim import kinematics as K
from grasprl.sim.expert import ScriptedExpert, sample_layout
from grasprl.sim.scene import Scene


@pytest.fixture(scope="module")
def scene():
    sc = Scene(make_renderer=False)
    yield sc
    sc.close()


def _run_expert(sc, seed, target="right", grip_bias=0.0, xy_offset=(0.0, 0.0)):
    """Play the scripted expert and report what happened to the target cube."""
    ex = ScriptedExpert(sc)
    layout = sample_layout(sc.cfg, np.random.default_rng(seed), target=target)
    if xy_offset != (0.0, 0.0):
        # Move the cube out from under the planned grasp, leaving the plan alone.
        plan = ex.plan(layout)
        key = "cube_right_xy" if target == "right" else "cube_left_xy"
        x, y = getattr(layout, key)
        setattr(layout, key, (x + xy_offset[0], y + xy_offset[1]))
    else:
        plan = ex.plan(layout)
    sc.reset(layout)
    sp = plan.setpoints.copy()
    sp[:, 5] = np.where(sp[:, 5] < 30, sp[:, 5] + grip_bias, sp[:, 5])
    gripped = lifted = 0
    for s in sp:
        sc.step(s, n_substeps=17)
        if sc.gripped_cube() == target:
            gripped += 1
        if sc.cube_height(target) > 0.05:
            lifted += 1
    for _ in range(40):
        sc.step(sp[-1], n_substeps=17)
    return {"gripped": gripped, "lifted": lifted, "in_cup": sc.cube_in_cup(target)}


def test_pads_are_positioned_on_the_measured_jaw_faces(scene):
    """The pads must sit exactly on the jaw faces, at every gripper command.

    They are authored from forward kinematics at the closed grip, so a
    regression here (a bad transform, or the compiler quietly ignoring the
    authored pose) shows up as a jaw gap that no longer tracks the command.
    """
    import mujoco

    m, d = scene.model, scene.data
    gid_f, gid_m = scene._pad_gids
    h = scene.grasp_cfg["pad_half_thickness"]
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    state = np.array(scene.cfg["init_pose"]["state"], float)

    def gap(grip):
        st = state.copy()
        st[5] = grip
        scene.set_arm_state(st)
        mujoco.mj_forward(m, d)
        R, p = d.xmat[gb].reshape(3, 3), d.xpos[gb]
        f = (R.T @ (d.geom_xpos[gid_f] - p))[0] + h
        mv = (R.T @ (d.geom_xpos[gid_m] - p))[0] - h
        return f, mv - f

    face, g19 = gap(scene.grasp_cfg["closed_grip"])
    assert face == pytest.approx(scene.grasp_cfg["fixed_face_x"], abs=1e-4)
    # A 30 mm cube in a 29.6 mm gap: a real interference fit, not a clearance.
    assert 0.028 < g19 < 0.030
    # And the gap must open with the command, or the gripper channel is inert.
    assert gap(25.0)[1] > g19 + 0.005
    assert gap(0.0)[1] < 0.006


def test_pads_touch_cubes_and_nothing_else(scene):
    """Masking must keep the pads off the table and the cup.

    The arm is non-colliding by design so the long fingers can dip to table
    height; the pads are the one exception, and if they inherit contact with the
    table or the cup that design breaks.
    """
    import mujoco

    m = scene.model
    pads = scene._pad_gids
    cube_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"cube_{s}")
                   for s in ("left", "right")}
    for gi in pads:
        for gj in range(m.ngeom):
            if gj in pads:
                continue
            collides = bool(m.geom_contype[gi] & m.geom_conaffinity[gj]) or \
                bool(m.geom_contype[gj] & m.geom_conaffinity[gi])
            if collides:
                assert m.geom_bodyid[gj] in cube_bodies, (
                    f"pad collides with geom {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gj)}"
                )


def test_a_good_grasp_holds_the_cube(scene):
    """The scripted expert, which grasps well, must be able to complete the task."""
    got = [_run_expert(scene, s) for s in range(4)]
    assert sum(r["in_cup"] for r in got) >= 3, got
    assert all(r["gripped"] > 0 for r in got), got


def test_an_over_open_gripper_drops_the_cube(scene):
    """Commanding the gripper a few units too open must lose the cube.

    The jaw gap runs 29.6 mm at command 19 and 37.7 mm at 25, so the cube is free
    well inside the range a policy's gripper channel wanders over. If this test
    stops failing the grasp, the gripper command has stopped mattering and there
    is nothing for either RL method to learn.
    """
    got = [_run_expert(scene, s, grip_bias=8.0) for s in range(4)]
    assert sum(r["in_cup"] for r in got) == 0, got


def test_a_misaligned_grasp_does_not_succeed(scene):
    """Closing 2 cm off the cube must not produce a successful pick."""
    got = [_run_expert(scene, s, xy_offset=(0.0, 0.02)) for s in range(4)]
    assert sum(r["in_cup"] for r in got) == 0, got


def test_weld_mode_still_available(scene):
    """``grasp.mode: weld`` must restore the original, never-slipping behaviour.

    It is the ablation that shows how much of the comparison's headroom is grasp
    physics rather than perception, so it has to keep working.
    """
    import copy

    cfg = copy.deepcopy(scene.cfg)
    cfg["grasp"]["mode"] = "weld"
    sc = Scene(cfg=cfg, make_renderer=False)
    try:
        assert sc.grasp_mode == "weld"
        assert not hasattr(sc, "_pad_gids")
        layout = sample_layout(sc.cfg, np.random.default_rng(0), target="right")
        sc.reset(layout)
        sc.attach("right")          # must not raise in weld mode
        sc.detach("right")
        with pytest.raises(RuntimeError):
            Scene(make_renderer=False).attach("right")   # contact mode: refuses
    finally:
        sc.close()


def test_gripper_torque_cap_is_applied(scene):
    """The calibrated servo torque must actually reach the actuator."""
    import mujoco

    fr = scene.grasp_cfg.get("gripper_forcerange")
    ai = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    if fr is None:
        return
    assert scene.model.actuator_forcerange[ai] == pytest.approx([-fr, fr])


def test_units_bridge_round_trips():
    """State -> ctrl -> state must be the identity, or sim and real disagree."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = np.array([rng.uniform(*K.REACHABLE_DEG[j]) for j in K.ARM_JOINTS]
                         + [rng.uniform(0, 100)], dtype=np.float32)
        back = K.qpos_to_state(K.state_to_ctrl(state))
        assert np.allclose(back, state, atol=1e-3)
