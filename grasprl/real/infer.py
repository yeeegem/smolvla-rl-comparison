"""Real-arm inference for the three arms of the comparison, on the SO-ARM101.

The control loop drives the same :class:`grasprl.policy.actor.Actor` the sim
evaluator uses. That is the point: ``base``, ``ppo`` and ``gaf`` then differ on
the real robot in exactly the ways they differ in sim -- same decision cadence
(``n_exec``), same deterministic ODE sampler, same seeded noise -- and any
difference in the scored result is attributable to the method rather than to two
subtly different inference paths.

It also solves a problem the stock LeRobot loop cannot: Guided Action Flow is not
expressible as ``policy.select_action``, because the critic has to intervene
*inside* the denoising loop. Driving the Actor directly sidesteps that entirely.

Observation and action plumbing is deliberately direct -- ``{motor}.pos`` in,
``{motor}.pos`` out, camera frames as uint8 arrays -- which matches the recorded
dataset's schema without needing dataset metadata on disk at eval time.
"""

from __future__ import annotations

import collections
import enum
import select
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

# Camera names in configs/eval_real.yaml, mapped onto the observation keys the
# policy was trained with.
CAMERA_KEYS = ("front", "wrist")


class StopReason(str, enum.Enum):
    """Why an episode ended."""

    OPERATOR = "operator"   # operator pressed Enter (task done or clearly failed)
    TIMEOUT = "timeout"     # hit max_steps without operator stop


def _enter_pressed() -> bool:
    """Non-blocking check for a newline on stdin (operator pressed Enter)."""
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


# ---------------------------------------------------------------------------
# Observation / action plumbing
# ---------------------------------------------------------------------------

def robot_observation(robot: SOFollower, motor_names: list[str]) -> dict:
    """One robot reading in the env's observation format."""
    obs = robot.get_observation()
    out = {"state": np.array([float(obs[f"{m}.pos"]) for m in motor_names],
                             dtype=np.float32)}
    for cam in CAMERA_KEYS:
        frame = obs.get(cam)
        if frame is None:
            raise KeyError(
                f"camera {cam!r} missing from the robot observation; "
                f"configs/eval_real.yaml must define infer.cameras.{cam}"
            )
        out[cam] = np.asarray(frame, dtype=np.uint8)
    return out


def robot_action(action: np.ndarray, motor_names: list[str]) -> dict:
    return {f"{m}.pos": float(v) for m, v in zip(motor_names, action, strict=True)}


def read_pose(robot: SOFollower, motor_names: list[str]) -> np.ndarray:
    """The arm's current joint positions in *motor_names* order."""
    obs = robot.get_observation()
    return np.array([float(obs[f"{m}.pos"]) for m in motor_names], dtype=np.float32)


def move_to_init(robot: SOFollower, motor_names: list[str], init_pose: np.ndarray,
                 fps: float, settle_steps: int = 45) -> None:
    """Drive the arm to a fixed init pose and hold it until it settles."""
    step_duration = 1.0 / fps
    action_dict = robot_action(init_pose, motor_names)
    for _ in range(settle_steps):
        t0 = time.perf_counter()
        robot.send_action(action_dict)
        remaining = step_duration - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)


def _dump_obs_frames(obs: dict, dump_dir: Path) -> None:
    """Save the camera images the policy is about to see, for identity checks."""
    from PIL import Image

    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    for cam in CAMERA_KEYS:
        path = dump_dir / f"{cam}.png"
        Image.fromarray(obs[cam]).save(path)
        print(f"  dumped {cam} {obs[cam].shape} {obs[cam].dtype} -> {path}")


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

def run_episode(
    actor,
    robot: SOFollower,
    motor_names: list[str],
    fps: float,
    max_steps: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    dump_frames_dir: Path | None = None,
) -> tuple[StopReason, float]:
    """Run one episode and return ``(stop_reason, duration_s)``.

    The actor is queried once per ``n_exec`` ticks and its chunk is played out at
    ``fps`` -- the same receding-horizon cadence the sim env uses, so the real
    numbers and the sim numbers describe the same controller.
    """
    if should_stop is None:
        should_stop = _enter_pressed

    step_duration = 1.0 / fps
    queue: collections.deque = collections.deque()
    latencies: collections.deque = collections.deque(maxlen=100)
    t_start = time.perf_counter()
    step_count = 0

    while True:
        step_t0 = time.perf_counter()
        obs = robot_observation(robot, motor_names)

        if dump_frames_dir is not None and step_count == 0:
            print(f"Dumping first-tick camera frames to {dump_frames_dir}:")
            _dump_obs_frames(obs, dump_frames_dir)

        if not queue:
            infer_t0 = time.perf_counter()
            chunk = actor.act([obs])[0]           # (n_exec, 6)
            latencies.append((time.perf_counter() - infer_t0) * 1000)
            arr = list(latencies)
            print(f"inference {arr[-1]:.1f} ms | mean {np.mean(arr):.1f} "
                  f"| p95 {np.percentile(arr, 95):.1f} | max {np.max(arr):.1f}")
            queue.extend(chunk)

        robot.send_action(robot_action(queue.popleft(), motor_names))
        step_count += 1

        if should_stop():
            return StopReason.OPERATOR, time.perf_counter() - t_start
        if max_steps is not None and step_count >= max_steps:
            return StopReason.TIMEOUT, time.perf_counter() - t_start

        remaining = step_duration - (time.perf_counter() - step_t0)
        if remaining > 0:
            time.sleep(remaining)


# ---------------------------------------------------------------------------
# Robot setup
# ---------------------------------------------------------------------------

def connect_robot(cfg: dict) -> tuple[SOFollower, list[str]]:
    """Connect to the SO-ARM101 follower and set the Feetech acceleration ramp.

    *cfg* is the parsed ``configs/eval_real.yaml``. Returns the connected robot
    and its motor names in state-vector order.
    """
    infer = cfg["infer"]
    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=cam["path"], width=cam["width"], height=cam["height"],
            fps=cam["fps"], fourcc=cam.get("fourcc"), backend=cam.get("backend", "auto"),
        )
        for name, cam in infer["cameras"].items()
    }
    robot = SOFollower(SOFollowerRobotConfig(
        port=infer["robot_port"], id=infer["robot_id"], cameras=cameras))
    robot.connect()
    motor_names = list(robot.bus.motors.keys())
    print(f"Robot connected. Motors: {motor_names}")

    accel = infer.get("motor_acceleration")
    if accel is not None:
        for motor in motor_names:
            robot.bus.write("Acceleration", motor, int(accel))
        print(f"Wrote Acceleration={int(accel)} to all {len(motor_names)} motors.")
    return robot, motor_names


def build_real_actor(method: str, checkpoint: str | Path, critic_dir: str | Path | None,
                     task: str, n_exec: int, guidance=None, seed: int = 0):
    """The same actor the sim evaluator builds, on the real robot's device."""
    from grasprl.policy.actor import build_actor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | method: {method} | checkpoint: {checkpoint}")
    if critic_dir:
        print(f"Critic: {critic_dir}")
    return build_actor(method, checkpoint, device, task, n_exec=n_exec,
                       critic_dir=critic_dir, guidance=guidance, seed=seed)
