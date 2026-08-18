"""Closed-loop pick-and-place environment for RL on a frozen SmolVLA policy.

One ``step`` is **one policy decision**: the policy emits a chunk and the env
executes ``n_exec`` of its actions open-loop at 30 Hz. Aligning the MDP with the
policy's real decision points is what makes a per-decision value function and a
per-decision advantage meaningful; stepping one control tick at a time would
give the critic 10x more states that the policy cannot actually react to.

Observations are exactly the imitation dataset's schema (``front`` / ``wrist``
uint8 frames plus the 6-D calibrated state), so the same frozen checkpoint runs
here, in ``grasprl.eval``, and on the real arm with no adaptation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar

import gymnasium as gym
import numpy as np

from grasprl.envs import rules
from grasprl.envs.reward import RewardConfig, Snapshot, potential, step_reward
from grasprl.sim import kinematics as K
from grasprl.sim.expert import sample_layout
from grasprl.sim.randomization import DomainRandomizer
from grasprl.sim.scene import Scene

os.environ.setdefault("MUJOCO_GL", "egl")

CONTROL_HZ = 30.0
N_SUBSTEPS = 17          # ~30 fps, matches how the demonstrations were recorded
DEFAULT_TASK = "Pick up a red cube and put it in the blue cup"


@dataclass
class EnvConfig:
    """Everything about an episode that is not scene geometry."""

    n_exec: int = 10            # actions executed per decision (of a 50-chunk)
    n_substeps: int = N_SUBSTEPS
    max_ticks: int = 300        # the scripted demos are 174 ticks long
    domain_randomize: bool = True
    render_size: tuple[int, int] | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)
    lift_height: float = 0.05
    grip_closed_below: float = 30.0


class PickPlaceEnv(gym.Env):
    """Two red cubes, one blue cup, pick either and drop it in.

    The task is deliberately **bimodal** (the cubes are identical, either is a
    valid target), which is why the mode-balance metric ``|P(left) - 0.5|`` is
    reported alongside success rate.
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: EnvConfig | None = None, scene_cfg: dict | None = None,
                 dr_cfg: dict | None = None, seed: int | None = None,
                 task: str = DEFAULT_TASK):
        self.cfg = cfg or EnvConfig()
        self.task = task
        self.scene = Scene(cfg=scene_cfg, render_size=self.cfg.render_size)
        self.dr = DomainRandomizer(self.scene, cfg=dr_cfg) if self.cfg.domain_randomize else None
        self._rng = np.random.default_rng(seed)

        lo, hi = rules.action_bounds(K)
        self.action_low = np.tile(lo, (self.cfg.n_exec, 1))
        self.action_high = np.tile(hi, (self.cfg.n_exec, 1))
        self.action_space = gym.spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float32)
        h = self.scene.cfg["cameras"]["height"]
        w = self.scene.cfg["cameras"]["width"]
        self.observation_space = gym.spaces.Dict({
            "front": gym.spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
            "wrist": gym.spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
            "state": gym.spaces.Box(lo, hi, (6,), dtype=np.float32),
        })
        self._state: rules.EpisodeState | None = None
        self._phi = 0.0
        self._ticks = 0

    # -- gym API -------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.dr is not None:
            self.dr.apply(self._rng)
        layout = sample_layout(self.scene.cfg, self._rng)
        self.scene.reset(layout)
        self._state = rules.EpisodeState(
            lift_height=self.cfg.lift_height,
            grip_closed_below=self.cfg.grip_closed_below,
        )
        self._phi = potential(self.scene, self._state, self.cfg.reward)
        self._ticks = 0
        self._captured = False
        self._layout = layout
        return self._observe(), {}

    def step(self, action: np.ndarray):
        """Execute one action chunk. ``action`` is ``(n_exec, 6)`` in LeRobot units."""
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(self.cfg.n_exec, 6),
                         self.action_low, self.action_high)
        before = Snapshot.of(self._state)
        for a in action:
            self.scene.step(a, n_substeps=self.cfg.n_substeps)
            # Acquisition assist, once per episode: see Scene.capture. Retention
            # from here on is contact physics, which is what is being measured.
            if not self._captured:
                side = self.scene.capture_candidate(float(a[5]))
                if side is not None:
                    self.scene.capture(side)
                    self._captured = True
            rules.update(self._state, self.scene, float(a[5]))
            self._ticks += 1

        phi = potential(self.scene, self._state, self.cfg.reward)
        reward = step_reward(self._phi, phi, self._state, before,
                             ticks=len(action), cfg=self.cfg.reward)
        self._phi = phi

        terminated = bool(self._state.success or rules.out_of_workspace(self.scene))
        truncated = bool(self._ticks >= self.cfg.max_ticks) and not terminated
        info = {"step_ticks": len(action)}
        if terminated or truncated:
            info["episode"] = self.episode_summary()
        return self._observe(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return np.concatenate([self.scene.render("front"), self.scene.render("wrist")], axis=1)

    def close(self):
        self.scene.close()

    # -- helpers -------------------------------------------------------------

    def _observe(self) -> dict:
        return {
            "front": self.scene.render("front"),
            "wrist": self.scene.render("wrist"),
            "state": self.scene.get_state().astype(np.float32),
        }

    def episode_summary(self) -> dict:
        """Per-episode record: the row that lands in the results table.

        ``cube_chosen`` is the cube that was actually lifted, which is what the
        real harness asks the operator for, and what the mode-balance metric
        counts.
        """
        s = self._state
        return {
            "success": bool(s.success),
            "category": rules.classify(s),
            "cube_chosen": s.lifted_side,
            "slips": int(s.slips),
            "ticks": int(self._ticks),
            "ever_gripped": bool(s.ever_gripped),
            "ever_lifted": bool(s.ever_lifted),
            "grasp_offset_m": float(s.grasp_offset_at_pickup),
            "grasp_yaw_err_rad": float(s.grasp_yaw_err_at_pickup),
        }
