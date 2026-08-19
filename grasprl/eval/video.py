"""Watch an arm of the comparison actually behave, with the grasp state overlaid.

Aggregate rates say *that* a policy fails; they never say *why*. The calibration
sweep left 55% of episodes in ``grabbed_nothing`` and no contact parameter moved
it, which is exactly the situation where you have to look at the rollout rather
than at another number.

Each frame is the front and wrist views side by side, annotated with the four
quantities that decide a grasp in this sim:

* **grip** -- the commanded gripper value, and the jaw gap it produces. The gap
  runs 29.6 mm at 19 and 32.4 mm at 21 against a 30 mm cube, so this single
  channel decides hold-versus-drop within about two units, and seeing it sit at
  25 while the arm is "grasping" explains an episode instantly.
* **gap to cube** -- distance from the grasp point to the nearer cube, which is
  what ``Scene.capture`` tests against ``grasp.capture_radius``.
* **grip state** -- whether both pads are actually loaded, and with how much
  normal force. This is the difference between touching a cube and holding one.
* **cube z** -- height above the table, so a lift and a drop are visible.

The banner turns green once a cube is held and red when a slip is recorded, and
the last frames of each episode carry its final category.

Usage::

    MUJOCO_GL=egl uv run python -m grasprl.eval.video --method base --episodes 4
    MUJOCO_GL=egl uv run python -m grasprl.eval.video --method gaf \\
        --critic runs/gaf_critic --episodes 4 --out results/gaf_rollout.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from grasprl.envs import rules
from grasprl.envs.pickplace_env import DEFAULT_TASK, EnvConfig, PickPlaceEnv
from grasprl.policy.actor import build_actor

# Colours are BGR because the frames are handed to OpenCV for drawing.
_WHITE = (255, 255, 255)
_GREY = (170, 170, 170)
_GREEN = (90, 220, 90)
_RED = (80, 80, 240)
_AMBER = (60, 190, 240)


def _jaw_gap(scene, grip_cmd: float) -> float:
    """Jaw gap (m) the commanded gripper value produces, from the pad poses."""
    import mujoco

    m, d = scene.model, scene.data
    if not hasattr(scene, "_pad_gids"):
        return float("nan")
    gid_f, gid_m = scene._pad_gids
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    R, p = d.xmat[gb].reshape(3, 3), d.xpos[gb]
    h = scene.grasp_cfg["pad_half_thickness"]
    face_f = (R.T @ (d.geom_xpos[gid_f] - p))[0] + h
    face_m = (R.T @ (d.geom_xpos[gid_m] - p))[0] - h
    return float(face_m - face_f)


# Header slots reserved on every frame. Fixed, not derived from the number of
# lines drawn: a shorter end-of-episode caption would otherwise produce a
# shorter frame, and the encoder needs every frame the same shape.
_HEADER_SLOTS = 5
_LINE_H = 26


def _annotate(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]],
              banner: tuple[str, tuple[int, int, int]] | None) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(frame)
    w = out.shape[1]
    header_h = _LINE_H * _HEADER_SLOTS + 8
    out = np.vstack([np.zeros((header_h, w, 3), np.uint8), out])

    for i, (text, colour) in enumerate(lines[:_HEADER_SLOTS]):
        cv2.putText(out, text, (10, 24 + _LINE_H * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, colour, 1, cv2.LINE_AA)
    if banner is not None and len(lines) < _HEADER_SLOTS:
        text, colour = banner
        cv2.putText(out, text, (10, 24 + _LINE_H * len(lines)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2, cv2.LINE_AA)
    cv2.line(out, (0, header_h - 1), (w, header_h - 1), (60, 60, 60), 1)
    return out


def record(
    method: str = "base",
    checkpoint: str = "checkpoints/base_smolvla",
    critic_dir: str | None = None,
    out: str = "results/rollout_base.mp4",
    episodes: int = 4,
    n_exec: int = 10,
    max_ticks: int = 300,
    domain_randomize: bool = True,
    seed: int = 0,
    fps: int = 30,
    scene_cfg: dict | None = None,
    task: str = DEFAULT_TASK,
    guidance=None,
) -> dict:
    import imageio
    import imageio.v3 as iio  # noqa: F401  (imported for the ffmpeg plugin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = build_actor(method, checkpoint, device, task, n_exec=n_exec,
                        critic_dir=critic_dir, guidance=guidance, seed=seed)
    env = PickPlaceEnv(
        cfg=EnvConfig(n_exec=n_exec, max_ticks=max_ticks,
                      domain_randomize=domain_randomize),
        scene_cfg=scene_cfg, task=task)
    scene = env.scene
    cube_size = scene.cfg["cubes"]["size"] * 2
    capture_r = scene.grasp_cfg.get("capture_radius")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, macro_block_size=1)
    n_frames = 0
    summaries: list[dict] = []
    try:
        for ep in range(episodes):
            actor.reset(seed + ep)
            obs, _ = env.reset(seed=seed + ep)
            slips_seen = 0
            done = False
            while not done:
                chunk = actor.act([obs])[0]
                for a in chunk:
                    # Step the scene directly so a frame can be drawn per control
                    # tick; env.step would advance the whole chunk at once and
                    # the grasp -- which happens inside a couple of ticks --
                    # would never be visible.
                    scene.step(a, n_substeps=env.cfg.n_substeps)
                    if not env._captured:
                        side = scene.capture_candidate(float(a[5]))
                        if side is not None:
                            scene.capture(side)
                            env._captured = True
                    rules.update(env._state, scene, float(a[5]))
                    env._ticks += 1

                    st = env._state
                    held = st.held
                    near, dist = rules.nearest_cube(scene)
                    gap = _jaw_gap(scene, float(a[5]))
                    force = scene.pad_contacts(held)[1] if held else 0.0
                    z = scene.cube_height(held or near)

                    grip_colour = _GREEN if gap < cube_size else _AMBER
                    near_colour = (_GREEN if capture_r and dist < capture_r else _GREY)
                    lines = [
                        (f"ep {ep}   tick {env._ticks:3d}/{max_ticks}", _WHITE),
                        (f"grip cmd {a[5]:5.1f}   jaw gap {gap * 1000:5.1f} mm "
                         f"(cube {cube_size * 1000:.0f} mm)", grip_colour),
                        (f"grasp point to {near:<5s} cube {dist * 100:5.2f} cm"
                         + (f"   capture < {capture_r * 100:.1f} cm" if capture_r else ""),
                         near_colour),
                        (f"held: {held or '-':<5s}  pad force {force:6.2f} N   "
                         f"cube z {z * 100:5.2f} cm", _GREEN if held else _GREY),
                    ]
                    banner = None
                    slips_seen = max(slips_seen, st.slips)
                    if st.success:
                        banner = ("SUCCESS", _GREEN)
                    elif slips_seen:
                        banner = (f"SLIP x{slips_seen}", _RED)
                    elif st.ever_lifted:
                        banner = ("carrying", _GREEN)
                    writer.append_data(_annotate(env.render(), lines, banner))
                    n_frames += 1

                    if st.success or rules.out_of_workspace(scene):
                        done = True
                        break
                    if env._ticks >= max_ticks:
                        done = True
                        break
                if not done:
                    obs = env._observe()

            summary = env.episode_summary()
            summaries.append({"episode": ep, **summary})
            colour = _GREEN if summary["success"] else _RED
            tail = (f"ep {ep}: {summary['category']}"
                    + (f"  ({summary['cube_chosen']})" if summary["cube_chosen"] else ""))
            # Hold the verdict on screen for a second so it is readable.
            for _ in range(fps):
                writer.append_data(_annotate(env.render(), [(tail, colour)], None))
                n_frames += 1
            print(f"ep {ep}: {summary['category']:16s} "
                  f"gripped={summary['ever_gripped']} lifted={summary['ever_lifted']} "
                  f"slips={summary['slips']} ticks={summary['ticks']}", flush=True)
    finally:
        env.close()
        writer.close()

    n = len(summaries)
    lifted = sum(s["ever_lifted"] for s in summaries)
    report = {
        "out": str(out), "method": method, "episodes": n, "frames": n_frames,
        "acquisition": lifted / n if n else 0.0,
        "retention": (sum(s["success"] for s in summaries) / lifted) if lifted else float("nan"),
        "categories": {c: sum(s["category"] == c for s in summaries)
                       for c in (rules.SUCCESS, *rules.CATEGORIES)},
        "episodes_detail": summaries,
    }
    print(f"\nwrote {out}  ({n_frames} frames, {n_frames / fps:.1f} s)")
    print(f"acquisition {report['acquisition']:.2f}  retention {report['retention']:.2f}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="base", choices=["base", "ppo", "gaf"])
    p.add_argument("--checkpoint", default="checkpoints/base_smolvla")
    p.add_argument("--critic", default=None, help="trained critic dir, for --method gaf")
    p.add_argument("--out", default=None, help="default results/rollout_<method>.mp4")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--n-exec", type=int, default=10)
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--no-domain-randomize", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args(argv)

    guidance = None
    if a.method == "gaf":
        import yaml

        from grasprl.gaf.guided_sampler import GuidanceConfig
        raw = yaml.safe_load(Path("configs/gaf.yaml").read_text()).get("guidance", {})
        guidance = GuidanceConfig(**raw)

    record(method=a.method, checkpoint=a.checkpoint, critic_dir=a.critic,
           out=a.out or f"results/rollout_{a.method}.mp4", episodes=a.episodes,
           n_exec=a.n_exec, max_ticks=a.max_ticks,
           domain_randomize=not a.no_domain_randomize, seed=a.seed, fps=a.fps,
           guidance=guidance)


if __name__ == "__main__":
    main()
