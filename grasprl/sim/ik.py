"""Damped-least-squares inverse kinematics for the 5-DOF SO-ARM101 arm.

The arm has 5 positioning joints (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll) plus the gripper. We solve a 6-D pose target (EE site
position + orientation) in the least-squares sense on those 5 joints: the target
is slightly over-constrained, so DLS finds the closest reachable pose, which is
exactly what a scripted top-down grasp wants (sub-centimetre position, an
approximately vertical approach).

The end-effector is the ``gripperframe`` site; its **local x-axis is the
approach (finger-pointing) direction**, so a top-down grasp asks for that axis to
point at world ``-z``.
"""

from __future__ import annotations

import numpy as np

# Arm = first 5 actuated joints; their qvel/dof indices are 0..4 and qpos 0..4
# (the gripper hinge is index 5, the free bodies come after).
ARM_DOF = slice(0, 5)


def grasp_orientation(approach_dir=(0.0, 0.0, -1.0), open_dir=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Desired EE rotation matrix (columns = site x,y,z axes in world).

    ``approach_dir`` is where the fingers point (site x); ``open_dir`` is the
    rough finger-opening direction (site y), re-orthogonalised against it.
    """
    x = np.asarray(approach_dir, float)
    x /= np.linalg.norm(x)
    y = np.asarray(open_dir, float)
    y = y - np.dot(y, x) * x
    if np.linalg.norm(y) < 1e-6:
        y = np.array([1.0, 0.0, 0.0]) - x[0] * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def solve_ik(
    model,
    data,
    site_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray | None = None,
    *,
    max_iters: int = 100,
    tol: float = 1e-3,
    damping: float = 1e-2,
    pos_weight: float = 1.0,
    ori_weight: float = 0.4,
    step_clip: float = 0.3,
) -> np.ndarray:
    """Return arm joint angles (5,) reaching ``target_pos``/``target_mat``.

    Iterates DLS on a scratch copy of the joint state (``data`` is mutated and
    forwarded during the solve; callers that care should pass a cloned ``data``
    or restore qpos afterwards). Joint limits are respected via clamping.
    """
    import mujoco

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    q_err = np.zeros(3)
    quat_cur = np.zeros(4)
    quat_des = np.zeros(4)
    if target_mat is not None:
        mujoco.mju_mat2Quat(quat_des, target_mat.reshape(9))

    lo = model.jnt_range[0:5, 0]
    hi = model.jnt_range[0:5, 1]

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        err = []
        p_err = (target_pos - data.site_xpos[site_id]) * pos_weight
        err.extend(p_err)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        rows = [jacp[:, ARM_DOF] * pos_weight]
        if target_mat is not None:
            mujoco.mju_mat2Quat(quat_cur, data.site_xmat[site_id])
            mujoco.mju_subQuat(q_err, quat_des, quat_cur)
            err.extend(q_err * ori_weight)
            rows.append(jacr[:, ARM_DOF] * ori_weight)
        e = np.asarray(err)
        if np.linalg.norm(e) < tol:
            break
        J = np.vstack(rows)
        # dq = J^T (J J^T + lambda^2 I)^-1 e
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(JJt.shape[0]), e)
        dq = np.clip(dq, -step_clip, step_clip)
        q = data.qpos[0:5] + dq
        data.qpos[0:5] = np.clip(q, lo, hi)

    return data.qpos[0:5].copy()
