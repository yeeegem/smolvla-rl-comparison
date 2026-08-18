"""Re-measure the SO-ARM101 jaw faces and check the pads land on them.

``configs/scene.yaml`` hard-codes two numbers taken from the vendored MJCF --
``fixed_face_x`` and ``moving_face_x``, the inner faces of the two jaws at the
closed grip. They are the anchor for the whole contact grasp, and nothing else
would notice if a future asset update moved them: the pads would simply sit in
mid-air and every grasp would fail for a reason that looks like a policy problem.

This script re-derives them from the mesh, prints the jaw gap against gripper
command, renders the closed grasp, and fails loudly if the config disagrees.

Usage::

    MUJOCO_GL=egl uv run python scripts/tune_jaw_pads.py [--out scene_views/grasp.png]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def measure(scene, grip: float) -> dict:
    """Jaw inner faces in the gripper frame, from the collision meshes."""
    import mujoco

    m, d = scene.model, scene.data
    state = np.array(scene.cfg["init_pose"]["state"], float)
    state[5] = grip
    scene.set_arm_state(state)
    mujoco.mj_forward(m, d)

    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    jb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    R, p = d.xmat[gb].reshape(3, 3), d.xpos[gb]

    def face(body_id, take_max: bool):
        best = None
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] != body_id or m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            did = m.geom_dataid[gi]
            a, n = m.mesh_vertadr[did], m.mesh_vertnum[did]
            v = m.mesh_vert[a:a + n].astype(float)
            Rg, pg = d.geom_xmat[gi].reshape(3, 3), d.geom_xpos[gi]
            P = (R.T @ (((Rg @ v.T).T + pg) - p).T).T
            band = P[(P[:, 2] > -0.115) & (P[:, 2] < -0.095)]   # the fingertip slab
            if len(band) == 0:
                continue
            x = band[:, 0].max() if take_max else band[:, 0].min()
            if best is None or (x > best[0] if take_max else x < best[0]):
                best = (x, float(np.abs(band[:, 1]).max()), float(P[:, 2].min()))
        return best

    fixed = face(gb, take_max=True)
    moving = face(jb, take_max=False)
    return {"grip": grip, "fixed_face_x": fixed[0], "moving_face_x": moving[0],
            "gap": moving[0] - fixed[0], "half_width": min(fixed[1], moving[1]),
            "tip_z": max(fixed[2], moving[2])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="render the closed grasp to this PNG")
    a = p.parse_args(argv)

    import mujoco

    from grasprl.sim.scene import Scene

    scene = Scene(make_renderer=a.out is not None)
    g = scene.grasp_cfg
    closed = float(g["closed_grip"])
    cube = 2 * scene.cfg["cubes"]["size"]

    print(f"Jaw gap vs gripper command (cube is {cube * 1000:.0f} mm):")
    for grip in (0.0, 10.0, closed, 21.0, 25.0, 42.0):
        m = measure(scene, grip)
        held = "holds" if m["gap"] < cube else "cube is free"
        print(f"  grip {grip:>5.1f} -> gap {m['gap'] * 1000:6.2f} mm   {held}")

    m = measure(scene, closed)
    print(f"\nAt the closed grip ({closed}):")
    print(f"  fixed jaw face   x = {m['fixed_face_x']:+.4f}  (config {g['fixed_face_x']:+.4f})")
    print(f"  moving jaw face  x = {m['moving_face_x']:+.4f}  (config {g['moving_face_x']:+.4f})")
    print(f"  jaw half-width   y = {m['half_width']:.4f}   (pad {g['pad_half_width']:.4f})")
    print(f"  fingertip depth  z = {m['tip_z']:.4f}")
    pad_bottom = g["pad_z"] - g["pad_half_height"]
    print(f"  pad spans z [{pad_bottom:.4f}, {g['pad_z'] + g['pad_half_height']:.4f}]")

    ok = True
    for key in ("fixed_face_x", "moving_face_x"):
        if abs(m[key] - g[key]) > 5e-4:
            print(f"\nMISMATCH: configs/scene.yaml {key} = {g[key]} but the asset "
                  f"measures {m[key]:.4f}. Update the config; the pads are anchored to it.")
            ok = False
    if pad_bottom < m["tip_z"] - 5e-4:
        print(f"\nWARNING: the pad extends {(m['tip_z'] - pad_bottom) * 1000:.1f} mm below "
              f"the fingertip, which is not a surface the real gripper has.")
    if ok:
        print("\nConfig matches the asset.")

    if a.out:
        state = np.array(scene.cfg["init_pose"]["state"], float)
        state[5] = closed
        scene.set_arm_state(state)
        mujoco.mj_forward(scene.model, scene.data)
        from PIL import Image

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(scene.render("wrist")).save(a.out)
        print(f"wrote {a.out}")
    scene.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
