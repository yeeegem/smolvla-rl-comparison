"""Scripted IK pick-and-place expert that seeds the bimodal distribution.

Given a :class:`~grasprl.sim.scene.Layout`, the expert plans a feed-forward
joint-space setpoint trajectory (in LeRobot units) by solving IK at a handful of
key poses and interpolating between them. A separate runner (the recorder)
applies the setpoints to the scene at 30 Hz and logs observations, so the expert
stays pure planning and is deterministic given a seed.

Bimodality: the *target* cube (left/right) is chosen by the layout sampler 50/50,
so a policy trained on the demos sees both modes in balance -- the whole point of
the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasprl.sim import ik
from grasprl.sim import kinematics as K
from grasprl.sim.scene import Layout, Scene

CONTROL_HZ = 30.0


@dataclass
class ExpertConfig:
    """Tunable constants for the scripted pick-and-place.

    Gripper values are in LeRobot RANGE_0_100 (0 = fully closed, 100 = fully
    open); the gap between the claws grows with the value. Heights (``*_z``) are
    the z of the grasp reference (the ``tcp`` fingertip site, see scene.py) in
    metres, table top at z=0. Durations are in 30 Hz control steps.
    """

    # -- gripper opening (RANGE_0_100) --------------------------------------
    grip_open: float = 42.0     # claws wide open while approaching/releasing
                                # (~70 mm gap); matches the dataset's open band
    grip_closed: float = 19.0   # holding a 3 cm cube: the moving jaw meets the
                                # cube's right face after the cube is snugged
                                # against the fixed jaw (~30 mm gap; see the tcp /
                                # grasp_snug sites in scene.py). Larger -> a visible
                                # gap on the moving side; smaller -> it clips.

    # -- waypoint heights (m; grasp reference = tcp fingertip site) ----------
    pregrasp_z: float = 0.075   # hover directly above the cube before descending
    grasp_z: float = 0.020      # fingertip at the cube (its centre is z=0.015);
                                # the whole gripper stays above the table here
    lift_z: float = 0.12        # straight up after grasping, clear of the table
    transport_z: float = 0.13   # carry height while moving over to the cup
    place_z: float = 0.10       # release height: above the cup rim (~0.075) so
                                # the claws drop the cube in without clipping/
                                # tipping the cup
    retreat_z: float = 0.15     # lift away after releasing, then return home

    # -- per-segment durations (30 Hz control steps) ------------------------
    reach: int = 28             # home -> pregrasp (above the cube)
    descend: int = 16           # pregrasp -> grasp (down onto the cube)
    grip_dwell: int = 12        # hold while the claws close (cube welded here)
    lift: int = 16              # grasp -> lift
    transport: int = 32         # lift -> over the cup
    place: int = 16             # descend toward the cup to the release height
    release_dwell: int = 12     # hold while the claws open (cube released here)
    retreat: int = 12           # release -> retreat up
    ret_home: int = 30          # retreat -> home pose


@dataclass
class Plan:
    """A planned episode: setpoints (T, 6) in LeRobot units + grasp events."""

    setpoints: np.ndarray
    target: str
    attach_step: int   # step at which to weld the target cube to the gripper
    detach_step: int   # step at which to release it (over the cup)


class ScriptedExpert:
    def __init__(self, scene: Scene, cfg: ExpertConfig | None = None):
        self.scene = scene
        self.cfg = cfg or ExpertConfig()
        import mujoco

        self._mj = mujoco
        self._sid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._home = np.array(scene.cfg["init_pose"]["state"], float)

    # -- IK at a world pose, warm-started from a scratch state ----------------

    def _ik_arm(self, scratch, pos, vertical: bool = True) -> np.ndarray:
        # Vertical top-down grasp: keep the claws pointing straight down (the
        # gripper's tool axis aligned with world -Z, i.e. the fingertip-site frame
        # aligned with world). This is a stable grasp and, because we target the
        # fingertip, it stays well above the table. ``vertical=False`` falls back
        # to position-only IK.
        target_mat = np.eye(3) if vertical else None
        return ik.solve_ik(
            self.scene.model, scratch, self._sid, np.asarray(pos, float),
            target_mat=target_mat, ori_weight=0.5 if vertical else 1e-6, damping=0.04,
        )

    def plan(self, layout: Layout) -> Plan:
        c = self.cfg
        cube_xy = layout.cube_left_xy if layout.target == "left" else layout.cube_right_xy
        cx, cy = cube_xy
        ux, uy = layout.cup_xy

        # A dedicated scratch MjData (NOT a deepcopy, which corrupts MjData) is
        # re-seeded at the home pose before each solve, so every keyframe picks
        # the same clean IK branch the standalone solver finds, instead of
        # drifting into contorted at-limit solutions.
        scratch = self._mj.MjData(self.scene.model)
        home_ctrl = K.state_to_ctrl(self._home)

        def arm_state(pos, grip, vertical=True):
            scratch.qpos[:] = 0.0
            scratch.qpos[:6] = home_ctrl
            self._mj.mj_forward(self.scene.model, scratch)
            q5 = self._ik_arm(scratch, pos, vertical)
            arm_deg = K.qpos_to_state(np.append(q5, 0.0))[:5]
            return np.append(arm_deg, grip).astype(np.float32)

        # Key setpoints (arm via position-only IK, gripper explicit). The weld
        # grasp makes orientation irrelevant, and position-only IK is the only
        # thing that reaches a low cube with this long gripper.
        # Grasp phases are VERTICAL (claws straight down) for a stable, non-slip
        # top-down grasp; carry/place phases use position-only IK so they can
        # reach the far cup.
        home = self._home.copy()
        pregrasp = arm_state((cx, cy, c.pregrasp_z), c.grip_open, vertical=True)
        grasp = arm_state((cx, cy, c.grasp_z), c.grip_open, vertical=True)
        grasp_closed = grasp.copy()
        grasp_closed[5] = c.grip_closed
        lift = arm_state((cx, cy, c.lift_z), c.grip_closed, vertical=True)
        transport = arm_state((ux, uy, c.transport_z), c.grip_closed, vertical=False)
        place = arm_state((ux, uy, c.place_z), c.grip_closed, vertical=False)
        release = place.copy()
        release[5] = c.grip_open
        retreat = arm_state((ux, uy, c.retreat_z), c.grip_open, vertical=False)

        # Stitch segments: (target_state, n_steps). Dwell = repeat same state.
        segs = [
            (pregrasp, c.reach),
            (grasp, c.descend),
            (grasp_closed, c.grip_dwell),
            (lift, c.lift),
            (transport, c.transport),
            (place, c.place),
            (release, c.release_dwell),
            (retreat, c.retreat),
            (home, c.ret_home),
        ]
        traj = _stitch(home, segs)
        # Weld the cube once the gripper has descended (start of grip dwell);
        # release it at the start of the release dwell, over the cup.
        attach_step = c.reach + c.descend
        detach_step = c.reach + c.descend + c.grip_dwell + c.lift + c.transport + c.place
        return Plan(traj, layout.target, attach_step, detach_step)


def _stitch(start: np.ndarray, segments) -> np.ndarray:
    """Linear interpolation through (target, n_steps) waypoints from ``start``."""
    out = []
    cur = start.copy()
    for target, n in segments:
        for k in range(1, n + 1):
            out.append(cur + (target - cur) * (k / n))
        cur = target.copy()
    return np.asarray(out, dtype=np.float32)


# -- layout sampling (50/50 bimodal) -----------------------------------------

def sample_layout(cfg: dict, rng: np.random.Generator, target: str | None = None) -> Layout:
    """Sample cube/cup positions and a left/right pick target.

    ``target`` forces which cube to pick (used to keep the recorded dataset
    exactly 50/50); if None it is a random coin flip.
    """
    cu, cp = cfg["cubes"], cfg["cup"]
    lx, ly = rng.uniform(*cu["x_range"]), rng.uniform(*cu["left_y_range"])
    rx, ry = rng.uniform(*cu["x_range"]), rng.uniform(*cu["right_y_range"])
    ux, uy = rng.uniform(*cp["x_range"]), rng.uniform(*cp["y_range"])
    yaw_range = cu.get("yaw_range", [0.0, 0.0])
    lyaw, ryaw = rng.uniform(*yaw_range), rng.uniform(*yaw_range)
    if target is None:
        target = "left" if rng.random() < 0.5 else "right"
    return Layout((lx, ly), (rx, ry), (ux, uy), target=target,
                  cube_left_yaw=float(lyaw), cube_right_yaw=float(ryaw))
