"""MuJoCo pick-and-place scene: SO-ARM101 + gray-blue table, white walls, two
red cubes, a procedural wall-ring blue cup, and front/wrist cameras.

Ported from ``sim2real-soarm-benchmark`` with one substantive change: a
**contact-physics grasp**. That repo makes the entire arm non-colliding and
holds the cube with an ``mjEQ_WELD`` equality, so a grasp can never slip -- and
grasp slip is the failure this repo exists to fix (30% of real trials). Here,
two thin high-friction *pad* geoms are glued to the jaw faces and masked to
collide only with the cubes. Everything else about the arm stays non-colliding,
so reaching behaviour, table clearance and the camera rig are unchanged, but
whether the cube stays in the gripper is now decided by real friction, real
squeeze force and real carry dynamics.

Set ``grasp.mode: weld`` in ``configs/scene.yaml`` to get the original
weld behaviour back (used by the ablation and by the sim2real demo recorder).

The arm comes from the vendored ``assets/so101/so101.xml``; everything else is
added with the MuJoCo ``MjSpec`` model-editing API from ``configs/scene.yaml``,
so the procedural cup (a bottom disk + N thin box wall segments forming a real
interior cavity) and the camera placement are all config-driven -- no XML string
templating.

Coordinate frame: arm base at the origin, table top at z=0, the arm reaches
toward +x. Left cube = +y, right cube = -y (a fixed labelling for the
mode-balance metric).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from grasprl.sim import kinematics as K

_REPO = Path(__file__).resolve().parents[2]
_SO101 = _REPO / "assets" / "so101" / "so101.xml"
_SCENE_CFG = _REPO / "configs" / "scene.yaml"


def _quat_lookat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """wxyz quat orienting a MuJoCo camera at ``pos`` to look at ``target``.

    MuJoCo cameras look along their local -Z with +Y up.
    """
    import mujoco

    pos = np.asarray(pos, float)
    target = np.asarray(target, float)
    z = pos - target  # local +z points away from the target
    n = np.linalg.norm(z)
    if n < 1e-9:
        raise ValueError("camera pos coincides with target")
    z /= n
    up = np.asarray(up, float)
    if abs(np.dot(z, up)) > 0.999:  # looking straight up/down -> pick another up
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1).reshape(9)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


@dataclass
class Layout:
    """One episode's object placement and which cube is the pick target."""

    cube_left_xy: tuple[float, float]
    cube_right_xy: tuple[float, float]
    cup_xy: tuple[float, float]
    target: str  # "left" or "right"
    cube_left_yaw: float = 0.0    # per-cube table rotation (radians)
    cube_right_yaw: float = 0.0


class Scene:
    """Loads the composed model and exposes reset / step / render helpers."""

    def __init__(self, cfg: dict | None = None, render_size: tuple[int, int] | None = None,
                 make_renderer: bool = True):
        import mujoco

        self.mj = mujoco
        self.cfg = cfg or yaml.safe_load(_SCENE_CFG.read_text())
        self.grasp_cfg = self.cfg.get("grasp", {})
        self.grasp_mode = self.grasp_cfg.get("mode", "contact")
        if self.grasp_mode not in ("contact", "weld"):
            raise ValueError(f"grasp.mode must be 'contact' or 'weld', got {self.grasp_mode!r}")
        self.model = self._build()
        K.apply_reachable_ranges(self.model)
        self._apply_collision_masks()
        self._recolor_arm()
        self.data = mujoco.MjData(self.model)
        if self.grasp_mode == "contact":
            self._pad_gids = tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in ("pad_fixed", "pad_moving")
            )
            self._apply_gripper_forcerange()

        # qpos addresses of each free-body joint (cubes, cup) for fast posing.
        self._free_qadr = {}
        for name in ("cube_left", "cube_right", "cup"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jadr = self.model.body_jntadr[bid]
            self._free_qadr[name] = self.model.jnt_qposadr[jadr]

        # Render options that hide site markers (the tcp/gripperframe sites would
        # otherwise show as little spheres in the images).
        self._scene_opt = mujoco.MjvOption()
        self._scene_opt.sitegroup[:] = 0

        # Offscreen renderer (EGL). Skipped for the interactive GLFW viewer,
        # which would conflict with an EGL context in the same process.
        self._renderer = None
        if make_renderer:
            h = render_size[0] if render_size else self.cfg["cameras"]["height"]
            w = render_size[1] if render_size else self.cfg["cameras"]["width"]
            if "MUJOCO_GL" not in os.environ:
                os.environ["MUJOCO_GL"] = "egl"
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)

    # -- model construction --------------------------------------------------

    def _build(self):
        mj = self.mj
        spec = mj.MjSpec.from_file(str(_SO101))
        wb = spec.worldbody
        c = self.cfg

        # Table: a box whose top face is at z=0.
        t = c["table"]
        wb.add_geom(
            name="table", type=mj.mjtGeom.mjGEOM_BOX,
            pos=[t["center_x"], 0.0, -t["thickness"] / 2],
            size=[t["half_x"], t["half_y"], t["thickness"] / 2],
            rgba=t["rgba"], friction=[1.0, 0.02, 0.001],
        )

        # White walls: front (far +x edge) + left/right (the near side is open).
        wcfg = c["walls"]
        wh, wt = wcfg["height"], wcfg["thickness"]
        wb.add_geom(
            name="wall_front", type=mj.mjtGeom.mjGEOM_BOX,
            pos=[wcfg["front_x"], 0.0, wh / 2],
            size=[wt, t["half_y"], wh / 2], rgba=wcfg["rgba"], contype=0, conaffinity=0,
        )
        for sgn, tag in ((1, "left"), (-1, "right")):
            wb.add_geom(
                name=f"wall_{tag}", type=mj.mjtGeom.mjGEOM_BOX,
                pos=[t["center_x"], sgn * wcfg["side_y"], wh / 2],
                size=[t["half_x"], wt, wh / 2], rgba=wcfg["rgba"], contype=0, conaffinity=0,
            )

        # Lights (type 1 == directional in mjtLightType; spotlights would be 0).
        for i, L in enumerate(c["lights"]):
            wb.add_light(name=f"light{i}", pos=L["pos"], dir=L["dir"], type=0)

        # Cubes (free bodies). Initial pose overwritten every reset().
        cu = c["cubes"]
        for name, y0 in (("cube_left", 0.10), ("cube_right", -0.10)):
            b = wb.add_body(name=name, pos=[0.25, y0, cu["z"]])
            b.add_freejoint()
            b.add_geom(
                type=mj.mjtGeom.mjGEOM_BOX, size=[cu["size"]] * 3, mass=cu["mass"],
                rgba=cu["rgba"], friction=cu["friction"], condim=4,
                solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
            )

        self._add_cup(wb)

        # Cameras.
        cam = c["cameras"]
        f = cam["front"]
        wb.add_camera(
            name="front", pos=f["pos"], quat=_quat_lookat(f["pos"], f["lookat"]),
            fovy=f["fovy"], resolution=[cam["width"], cam["height"]],
        )
        wcam = cam["wrist"]
        parent = spec.body(wcam["parent_body"])
        wpos = np.asarray(wcam["pos"], float)
        # Prefer an explicit local quat (community/official camera definitions
        # give one); otherwise aim it via a local look-at point.
        if "quat" in wcam:
            wquat = np.asarray(wcam["quat"], float)
        else:
            wquat = _quat_lookat(wpos, wcam["lookat_local"])
        parent.add_camera(
            name="wrist", pos=wpos, quat=wquat,
            fovy=wcam["fovy"], resolution=[cam["width"], cam["height"]],
        )
        # Real white printed hex-nut camera mount (official SO101 STL, mm -> m)
        # attached to the gripper, with the black camera module at the lens.
        # Its transform (mount.pos/quat in the gripper local frame) is config-
        # driven so it can be dialled in to clamp the wrist correctly.
        # Wrap the mount mesh in its own child body so its pose is authored
        # cleanly (MuJoCo recenters mesh *geoms* to their CoM, but not body
        # frames) -- this makes scripts/tune_mount.py's printed body pose paste
        # straight back into mount.pos/quat.
        mcfg = wcam.get("mount", {})
        spec.add_mesh(name="wrist_cam_mount_mesh", file="wrist_cam_mount.stl",
                      scale=[0.001, 0.001, 0.001])
        mount_body = parent.add_body(
            name="wrist_cam_mount_body",
            pos=mcfg.get("pos", [0.0, 0.045, 0.0]),
            quat=mcfg.get("quat", [0.7071, -0.7071, 0.0, 0.0]),
        )
        mount_body.add_geom(
            name="wrist_cam_mount", type=mj.mjtGeom.mjGEOM_MESH,
            meshname="wrist_cam_mount_mesh",
            rgba=[0.90, 0.90, 0.92, 1.0], contype=0, conaffinity=0, group=2,
        )
        parent.add_geom(
            name="wrist_cam_module", type=mj.mjtGeom.mjGEOM_BOX,
            pos=wpos.tolist(), quat=wquat.tolist(),
            size=[0.012, 0.012, 0.008], rgba=[0.05, 0.05, 0.05, 1.0],
            contype=0, conaffinity=0, group=2,
        )

        # Two grasp-reference sites, both at the fingertip depth (z=-0.105) so
        # the whole gripper stays above the table (a point between the finger
        # bases would drive the long fingers ~65 mm through the table).
        #
        # On the real SO-ARM101 only the moving jaw actuates; the fixed jaw's
        # inner face is stationary at local x=-0.0079. So we model a real grasp:
        #  - `tcp` (x=0.015): the IK/approach target. With the cube here the
        #    fixed jaw has ~8 mm clearance, so it won't hit the cube on descent.
        #  - `grasp_snug` (x=0.007): where the moving jaw pushes the cube -- its
        #    left face against the fixed jaw. attach() welds the cube here, so at
        #    grasp the cube moves over to the fixed jaw and both claws touch it.
        spec.body("gripper").add_site(name="tcp", pos=[0.015, 0.0, -0.105])
        spec.body("gripper").add_site(name="grasp_snug", pos=[0.007, 0.0, -0.105])


        # Inactive welds used to model a reliable grasp (see attach/detach):
        # frictional grasping of a 3 cm cube is unreliable with this gripper's
        # bulky meshes, so the expert welds the cube to the gripper at grasp.
        for cube in ("cube_left", "cube_right"):
            eq = spec.add_equality(
                type=mj.mjtEq.mjEQ_WELD, name=f"weld_{cube}",
                name1="gripper", name2=cube, active=False,
            )
            eq.objtype = mj.mjtObj.mjOBJ_BODY

        if self.grasp_mode != "contact":
            return spec.compile()

        # Two-pass compile. The pads must sit on the *jaw faces*, and the moving
        # jaw's face pose is only knowable from forward kinematics at the closed
        # grip -- which needs a compiled model. Authoring them at a placeholder
        # pose and fixing up model.geom_pos afterwards does not work: the
        # broadphase BVH is built at compile time from the authored poses, so a
        # pad moved 10 cm away is pruned before narrowphase and silently never
        # collides. So: compile once to measure, then author and compile again.
        probe = spec.compile()
        K.apply_reachable_ranges(probe)
        fixed_pose, moving_pose = self._measure_pad_poses(probe)
        self._add_jaw_pads(spec, fixed_pose, moving_pose)
        return spec.compile()

    # -- contact grasp: jaw pads ---------------------------------------------

    def _measure_pad_poses(self, probe):
        """Forward-kinematic the jaws at the closed grip and return each pad's
        pose *in its parent body frame*, as ``(pos, quat)`` pairs.

        ``pad_fixed`` hangs off the ``gripper`` body, so its target is already a
        gripper-frame coordinate. ``pad_moving`` hangs off the rotating jaw, so
        its gripper-frame target is transformed through the jaw frame at that
        hinge angle.
        """
        mj = self.mj
        g = self.grasp_cfg
        h, z = g["pad_half_thickness"], g["pad_z"]

        d = mj.MjData(probe)
        state = np.array(self.cfg["init_pose"]["state"], float)
        state[5] = g["closed_grip"]
        d.qpos[:6] = K.state_to_ctrl(state)
        mj.mj_forward(probe, d)

        gb = mj.mj_name2id(probe, mj.mjtObj.mjOBJ_BODY, "gripper")
        jb = mj.mj_name2id(probe, mj.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        R_g, p_g = d.xmat[gb].reshape(3, 3), d.xpos[gb]
        R_j, p_j = d.xmat[jb].reshape(3, 3), d.xpos[jb]

        # Each pad's centre, placed so its inner face lands on its jaw's face.
        fixed_pos = np.array([g["fixed_face_x"] - h, 0.0, z])
        moving_grip = np.array([g["moving_face_x"] + h, 0.0, z])

        moving_pos = R_j.T @ (R_g @ moving_grip + p_g - p_j)
        moving_quat = np.zeros(4)
        mj.mju_mat2Quat(moving_quat, (R_j.T @ R_g).reshape(9))
        return (fixed_pos, np.array([1.0, 0.0, 0.0, 0.0])), (moving_pos, moving_quat)

    def _add_jaw_pads(self, spec, fixed_pose, moving_pose):
        """Add the two contact pads that make a frictional grasp possible.

        Because the jaw *meshes* are non-colliding (see
        :meth:`_apply_collision_masks`), these two boxes are the gripper's
        entire contact geometry: their size, placement and friction *are* the
        grasp model.
        """
        mj = self.mj
        g = self.grasp_cfg
        size = [g["pad_half_thickness"], g["pad_half_width"], g["pad_half_height"]]
        for name, body, (pos, quat) in (
            ("pad_fixed", "gripper", fixed_pose),
            ("pad_moving", "moving_jaw_so101_v1", moving_pose),
        ):
            spec.body(body).add_geom(
                name=name, type=mj.mjtGeom.mjGEOM_BOX, size=size,
                pos=pos.tolist(), quat=quat.tolist(), mass=0.0,
                rgba=[0.15, 0.15, 0.17, 1.0], group=2,
                friction=g["pad_friction"], condim=4,
                solref=g["pad_solref"], solimp=g["pad_solimp"],
            )

    def _apply_gripper_forcerange(self) -> None:
        """Cap the gripper servo torque at the configured value.

        The vendored MJCF ships the STS3215 stall torque (3.35 Nm), which at the
        jaw's ~40 mm lever is ~84 N of pinch -- so much that a geometrically
        sound grasp could never slip and the sim would under-report the very
        failure we are studying. The real servo delivers far less under a
        sustained squeeze and the printed jaws flex. Calibrated in
        :mod:`grasprl.sim.calibrate`.
        """
        fr = self.grasp_cfg.get("gripper_forcerange")
        if fr is None:
            return
        ai = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.model.actuator_forcerange[ai] = (-abs(float(fr)), abs(float(fr)))

    def _apply_collision_masks(self):
        """Filter contacts so the slim fingers can reach low cubes without the
        bulky arm/gripper meshes false-colliding with the table.

        Symmetric ``contype == conaffinity`` masks: two geoms collide iff their
        masks share a bit. One bit per desired contact edge:

            bit0  table-cube
            bit1  table-cup, cube-cup
            bit2  cube-cube
            bit3  pad-cube        <- new, the contact grasp

            TABLE = 0b0011   CUBE = 0b1111   CUP = 0b0110
            PAD   = 0b1000   arm  = 0 (collides with nothing)

        The pads therefore touch cubes and *only* cubes: ``PAD & TABLE == 0`` and
        ``PAD & CUP == 0``, so a pad can never catch on the table or knock the
        cup, while the rest of the arm keeps the original free pass that lets it
        dip to table height.
        """
        mj, m = self.mj, self.model

        def bid(n):
            return mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)

        TABLE, CUBE, CUP, PAD, NONE = 0b0011, 0b1111, 0b0110, 0b1000, 0
        PADS = {"pad_fixed", "pad_moving"}
        cubes = {bid("cube_left"), bid("cube_right")}
        cup = bid("cup")
        arm = {bid(n) for n in
               ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                "moving_jaw_so101_v1")}
        for gi in range(m.ngeom):
            b = m.geom_bodyid[gi]
            gname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, gi)
            if gname in PADS:      # checked before `arm`: the pads live on arm bodies
                mask = PAD
            elif gname == "table":  # table geom lives on the worldbody
                mask = TABLE
            elif b in cubes:
                mask = CUBE
            elif b == cup:
                mask = CUP
            elif b in arm:
                mask = NONE
            else:
                continue  # walls etc. already non-colliding
            m.geom_contype[gi] = mask
            m.geom_conaffinity[gi] = mask

    def _recolor_arm(self):
        """Recolour the vendored MJCF's printed parts (yellow) to the configured
        arm colour, matching the real white-printed SO-ARM101. Dark motor
        materials are left untouched."""
        m = self.model
        rgba = self.cfg.get("robot", {}).get("arm_rgba")
        if rgba is None:
            return
        for i in range(m.nmat):
            r, g, b = m.mat_rgba[i, :3]
            if r > 0.6 and g > 0.5 and b < 0.4:  # the yellow printed-part material
                m.mat_rgba[i, :3] = rgba[:3]

    def _add_cup(self, wb):
        mj = self.mj
        cp = self.cfg["cup"]
        b = wb.add_body(name="cup", pos=[0.27, 0.0, 0.0])
        b.add_freejoint()
        inner, wallt, h = cp["inner_radius"], cp["wall_thickness"], cp["height"]
        bt = cp["bottom_thickness"]
        outer = inner + wallt
        # Bottom disk (heavy + wide base -> stays put but tippable).
        b.add_geom(
            name="cup_bottom", type=mj.mjtGeom.mjGEOM_CYLINDER,
            pos=[0, 0, bt / 2], size=[outer, bt / 2, 0], mass=cp["bottom_mass"],
            rgba=cp["rgba"], friction=cp["friction"], condim=3,
        )
        # Wall ring: N thin boxes around a circle, slight tangential overlap.
        n = int(cp["n_segments"])
        r_wall = inner + wallt / 2
        chord_half = r_wall * math.tan(math.pi / n)
        for i in range(n):
            th = 2 * math.pi * i / n
            b.add_geom(
                name=f"cup_wall_{i}", type=mj.mjtGeom.mjGEOM_BOX,
                pos=[r_wall * math.cos(th), r_wall * math.sin(th), bt + h / 2],
                quat=_z_quat(th), size=[wallt / 2, chord_half, h / 2],
                mass=cp["wall_mass"], rgba=cp["rgba"], friction=cp["friction"], condim=3,
            )

    # -- episode control -----------------------------------------------------

    def _set_free_pose(self, name: str, xy, z: float, yaw: float = 0.0):
        adr = self._free_qadr[name]
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], z]
        self.data.qpos[adr + 3 : adr + 7] = _z_quat(yaw)

    def set_arm_state(self, state: np.ndarray, settle: bool = False):
        """Set the arm to a LeRobot-unit state (instant) and its actuator target."""
        ctrl = K.state_to_ctrl(state)
        self.data.qpos[:6] = ctrl
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = ctrl
        self.mj.mj_forward(self.model, self.data)

    def reset(self, layout: Layout, init_state: np.ndarray | None = None):
        self.mj.mj_resetData(self.model, self.data)
        cu = self.cfg["cubes"]
        self._set_free_pose("cube_left", layout.cube_left_xy, cu["z"], layout.cube_left_yaw)
        self._set_free_pose("cube_right", layout.cube_right_xy, cu["z"], layout.cube_right_yaw)
        self._set_free_pose("cup", layout.cup_xy, 0.0)
        state = init_state if init_state is not None else np.array(self.cfg["init_pose"]["state"], float)
        self.set_arm_state(state)
        self.mj.mj_forward(self.model, self.data)

    def step(self, ctrl_state: np.ndarray, n_substeps: int = 1):
        """Command the arm with a LeRobot-unit target and advance the sim."""
        self.data.ctrl[:6] = K.state_to_ctrl(ctrl_state)
        for _ in range(n_substeps):
            self.mj.mj_step(self.model, self.data)

    def get_state(self) -> np.ndarray:
        """Current arm state in LeRobot units (degrees + RANGE_0_100 gripper)."""
        return K.qpos_to_state(self.data.qpos[:6])

    def attach(self, cube: str):
        """Weld ``cube`` ('left'/'right') to the gripper for a reliable, precise
        grasp. The cube is first snapped to the grasp point (the ``tcp`` site,
        between the fingers) so the grasp is exact regardless of position-control
        tracking lag, then welded at that relative pose."""
        if self.grasp_mode != "weld":
            raise RuntimeError(
                "attach() is the weld-grasp abstraction and is only available with "
                "grasp.mode='weld'; in contact mode the cube is held by friction."
            )
        mj, m, d = self.mj, self.model, self.data
        eid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_EQUALITY, f"weld_cube_{cube}")
        g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gripper")
        cb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"cube_{cube}")

        # Snap the cube onto the snug grasp point (against the fixed jaw) so it
        # moves over to the fixed jaw as the moving jaw closes -- a precise,
        # realistic grasp.
        snug = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "grasp_snug")
        adr = self._free_qadr[f"cube_{cube}"]
        d.qpos[adr : adr + 3] = d.site_xpos[snug]
        dof = m.body_dofadr[cb]
        d.qvel[dof : dof + 6] = 0.0
        mj.mj_forward(m, d)

        R1 = d.xmat[g].reshape(3, 3)
        relpos = R1.T @ (d.xpos[cb] - d.xpos[g])
        q1 = np.zeros(4)
        q1c = np.zeros(4)
        relq = np.zeros(4)
        mj.mju_mat2Quat(q1, d.xmat[g])
        mj.mju_negQuat(q1c, q1)
        mj.mju_mulQuat(relq, q1c, d.xquat[cb])
        m.eq_data[eid, 0:3] = 0.0          # anchor (unused with relpose)
        m.eq_data[eid, 3:6] = relpos       # relative position
        m.eq_data[eid, 6:10] = relq        # relative orientation
        m.eq_data[eid, 10] = 1.0           # torquescale
        d.eq_active[eid] = 1

    def detach(self, cube: str):
        """Release a welded cube."""
        if self.grasp_mode != "weld":
            raise RuntimeError("detach() requires grasp.mode='weld'")
        eid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_EQUALITY, f"weld_cube_{cube}")
        self.data.eq_active[eid] = 0

    def render(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera, scene_option=self._scene_opt)
        return self._renderer.render()

    def close(self):
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- geometry helpers used by the expert / success check -----------------

    def body_xpos(self, name: str) -> np.ndarray:
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def ee_xpos(self) -> np.ndarray:
        sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, "gripperframe")
        return self.data.site_xpos[sid].copy()

    def grasp_xpos(self) -> np.ndarray:
        """World position of the grasp point (the centre of the closed jaw gap).

        This, not ``tcp``, is where a cube ends up once the moving jaw has pushed
        it home, so it is the reference every grasp-geometry measurement uses.
        """
        sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, "grasp_snug")
        return self.data.site_xpos[sid].copy()

    def tcp_xpos(self) -> np.ndarray:
        sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, "tcp")
        return self.data.site_xpos[sid].copy()

    def cube_in_cup(self, cube: str) -> bool:
        """True if the named cube ('left'/'right') rests inside the cup cavity."""
        cube_p = self.body_xpos(f"cube_{cube}")
        cup_p = self.body_xpos("cup")
        cp = self.cfg["cup"]
        radial = math.hypot(cube_p[0] - cup_p[0], cube_p[1] - cup_p[1])
        z_rel = cube_p[2] - cup_p[2]
        inside_xy = radial < (cp["inner_radius"] - self.cfg["cubes"]["size"])
        inside_z = self.cfg["cubes"]["size"] < z_rel < cp["height"]
        return bool(inside_xy and inside_z)


    # -- contact-grasp introspection (used by the env + failure classifier) ---

    def pad_contacts(self, cube: str) -> tuple[int, float]:
        """``(n_contacts, total_normal_force)`` between the jaw pads and ``cube``.

        The normal force is what decides whether friction can hold the cube, so
        it is the natural "is this actually gripped" signal -- far more reliable
        than a proximity test, which is all the weld-based sim could offer.
        """
        if self.grasp_mode != "contact":
            return (0, 0.0)
        mj, m, d = self.mj, self.model, self.data
        cb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"cube_{cube}")
        pads = set(self._pad_gids)
        n, total = 0, 0.0
        ft = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pad_hit = (g1 in pads) != (g2 in pads)
            if not pad_hit:
                continue
            other = g2 if g1 in pads else g1
            if m.geom_bodyid[other] != cb:
                continue
            mj.mj_contactForce(m, d, i, ft)
            n += 1
            total += abs(float(ft[0]))    # normal component, contact frame
        return (n, total)

    def capture(self, cube: str) -> None:
        """Cancel the *lateral* placement error of a grasp, then let physics decide.

        Measured on this checkpoint (scripts/, and the README's "What the sim can
        and cannot model"): in MuJoCo the frozen policy closes its jaws with up
        to ~20 mm of across-the-jaws error, where the scripted expert manages
        0.5 mm. A 30 mm cube in a 29.6 mm jaw gap cannot survive that, so a
        faithful pinch misses ~70% of the time -- while the same policy grasps
        83% of the time on the real arm. The error is a sim perception artifact,
        not a property of the grasp, and no contact parameter fixes it: pad size,
        friction, servo torque and contact softness were all swept and none
        moved the acquisition rate.

        So acquisition is abstracted and *retention* is left to real physics.
        This nudges the cube along the gripper's y axis (the one direction the
        jaws cannot self-correct -- the moving jaw already pushes the cube home
        along x) and leaves everything that reflects the policy's actual
        behaviour untouched:

          * the grasp **height** -- close high and the jaws catch only the top
            few millimetres, exactly as they would on the real arm;
          * the cube's **yaw** -- meet a corner instead of a face and it squirts
            out;
          * the **gripper command** -- the jaw gap runs 29.6 mm at 19 and
            32.4 mm at 21, so a couple of units too open still drops the cube.

        Those are the variables a policy can learn to fix, and they are what both
        RL methods are being asked to improve.
        """
        mj, m, d = self.mj, self.model, self.data
        gb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gripper")
        R = d.xmat[gb].reshape(3, 3)
        p_cube = self.body_xpos(f"cube_{cube}")
        offset = R.T @ (p_cube - self.grasp_xpos())
        correction = R @ np.array([0.0, -offset[1], 0.0])   # y only

        adr = self._free_qadr[f"cube_{cube}"]
        d.qpos[adr : adr + 3] = p_cube + correction
        cb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"cube_{cube}")
        dof = m.body_dofadr[cb]
        d.qvel[dof : dof + 6] = 0.0
        mj.mj_forward(m, d)

    def capture_candidate(self, grip_cmd: float) -> str | None:
        """Which cube, if any, is close enough to be captured on this tick.

        ``None`` unless the gripper is closing and a cube is inside
        ``grasp.capture_radius`` of the grasp point -- so a policy that closes on
        empty space still gets ``grabbed_nothing``, which is 13% of the real
        arm's trials and must not be abstracted away too.
        """
        g = self.grasp_cfg
        radius = g.get("capture_radius")
        if radius is None or self.grasp_mode != "contact":
            return None
        if grip_cmd >= g.get("capture_grip_below", 30.0):
            return None
        p = self.grasp_xpos()
        best, best_d = None, float(radius)
        for side in ("left", "right"):
            dist = float(np.linalg.norm(p - self.body_xpos(f"cube_{side}")))
            if dist < best_d:
                best, best_d = side, dist
        return best

    def gripped_cube(self, min_force: float = 0.05) -> str | None:
        """Which cube is currently pinched by *both* pads, if any.

        Both pads must be in contact: a cube merely leaning on one jaw is not a
        grasp, and counting it as one is what turns a genuine slip into a
        phantom success.
        """
        if self.grasp_mode != "contact":
            return None
        mj, m, d = self.mj, self.model, self.data
        gid_fixed, gid_moving = self._pad_gids
        for side in ("left", "right"):
            cb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"cube_{side}")
            hit = {gid_fixed: False, gid_moving: False}
            force = 0.0
            ft = np.zeros(6)
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                pad = g1 if g1 in hit else (g2 if g2 in hit else None)
                if pad is None:
                    continue
                other = g2 if pad == g1 else g1
                if m.geom_bodyid[other] != cb:
                    continue
                hit[pad] = True
                mj.mj_contactForce(m, d, i, ft)
                force += abs(float(ft[0]))
            if all(hit.values()) and force >= min_force:
                return side
        return None

    def cube_height(self, cube: str) -> float:
        """Cube centre height above the table top (table top is z=0)."""
        return float(self.body_xpos(f"cube_{cube}")[2])

    def grasp_offset(self, cube: str) -> float:
        """Horizontal distance from the grasp point to the cube centre (m).

        Measured against ``grasp_snug``, which sits exactly at the centre of the
        jaw gap at the closed grip -- so this is "how far off-centre is the cube
        in the gripper", the geometric precursor of a slip.
        """
        p = self.grasp_xpos()
        c = self.body_xpos(f"cube_{cube}")
        return float(np.hypot(p[0] - c[0], p[1] - c[1]))

    def grasp_yaw_err(self, cube: str) -> float:
        """Smallest angle (rad) between the pinch axis and a cube face normal.

        A cube has 90-degree symmetry, so this is in ``[0, pi/4]``: 0 means the
        jaws meet two flat faces, pi/4 means they are pinching opposite corners
        -- the classic way a top-down grasp squirts the cube out.
        """
        mj, m, d = self.mj, self.model, self.data
        gb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gripper")
        cb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"cube_{cube}")
        pinch = d.xmat[gb].reshape(3, 3)[:, 0]        # gripper local +x
        face = d.xmat[cb].reshape(3, 3)[:, 0]         # cube local +x
        pinch = pinch[:2] / max(np.linalg.norm(pinch[:2]), 1e-9)
        face = face[:2] / max(np.linalg.norm(face[:2]), 1e-9)
        ang = math.acos(float(np.clip(abs(np.dot(pinch, face)), 0.0, 1.0)))
        return min(ang, math.pi / 2 - ang)

    def cup_tilt(self) -> float:
        """Angle (rad) between the cup's axis and world +z. Large => knocked over."""
        cb = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "cup")
        axis = self.data.xmat[cb].reshape(3, 3)[:, 2]
        return math.acos(float(np.clip(axis[2], -1.0, 1.0)))


def _z_quat(yaw: float) -> np.ndarray:
    """wxyz quat for a rotation of ``yaw`` radians about +Z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
