# SO-ARM101 asset (vendored)

Vendored from **TheRobotStudio/SO-ARM100**, path `Simulation/SO101/`
(commit in `SOURCE_COMMIT.txt`). License: see `LICENSE` (Apache-2.0). Citation: `CITATION.cff`.

- `so101.xml` — official onshape-to-robot MuJoCo MJCF (`so101_new_calib.xml`), the **new-calibration**
  variant (calibration offsets already baked into the joint ranges). No URDF→MJCF conversion was needed.
- `so101.urdf` — reference URDF (`so101_new_calib.urdf`), not used at runtime.
- `joints_properties.xml` — sts3215 servo defaults (kp/damping/armature) referenced by the MJCF.
- `assets/*.stl` — link meshes (`meshdir="assets"` inside `so101.xml`).

Key facts the rest of the repo depends on:
- Joints (in order): `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` —
  identical names + order to the real dataset's `observation.state` / `action`.
- 6 `position` actuators (one per joint), `angle="radian"`.
- Gripper hinge range ≈ −10°→100° → maps onto LeRobot's `RANGE_0_100` gripper convention.
- End-effector site `gripperframe` (on the `gripper` body) — used as the IK target frame.

The scene that adds the table, walls, cubes, cup, and cameras is `assets/scene.xml`, which `<include>`s
this file. The MJCF radians are converted to the real arm's calibrated degrees / `RANGE_0_100` units by
`sim2real_soarm/sim/kinematics.py`.
