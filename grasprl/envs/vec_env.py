"""In-process vectorisation of :class:`PickPlaceEnv`.

A plain Python loop over N envs, not subprocesses. Rollouts here are bottlenecked
by the SmolVLA forward pass on one GPU, not by MuJoCo, so parallel processes buy
nothing and would each need their own EGL context. One process, one context, N
scenes.

Auto-resets on episode end and hands the terminal observation back in
``info["final_obs"]`` so the trainer can bootstrap a truncated episode's value
instead of silently treating the time limit as a real terminal state.
"""

from __future__ import annotations

import numpy as np

from grasprl.envs.pickplace_env import DEFAULT_TASK, EnvConfig, PickPlaceEnv


class VecPickPlaceEnv:
    """N independent :class:`PickPlaceEnv` stepped together."""

    def __init__(self, n_envs: int, cfg: EnvConfig | None = None,
                 scene_cfg: dict | None = None, dr_cfg: dict | None = None,
                 seed: int = 0, task: str = DEFAULT_TASK):
        self.n_envs = n_envs
        self.envs = [
            PickPlaceEnv(cfg=cfg, scene_cfg=scene_cfg, dr_cfg=dr_cfg,
                         seed=seed + 1000 * i, task=task)
            for i in range(n_envs)
        ]
        self.cfg = self.envs[0].cfg
        self.task = task

    def reset(self, seed: int | None = None) -> list[dict]:
        return [e.reset(seed=None if seed is None else seed + 1000 * i)[0]
                for i, e in enumerate(self.envs)]

    def step(self, actions: np.ndarray):
        """``actions`` is ``(n_envs, n_exec, 6)``. Returns SB3-style batched output."""
        obs, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            done = term or trunc
            if done:
                info["truncated"] = bool(trunc and not term)
                info["final_obs"] = o
                info["final_info"] = info.get("episode")
                o, _ = env.reset()
            obs.append(o)
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return obs, np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos

    def render(self, index: int = 0) -> np.ndarray:
        return self.envs[index].render()

    def close(self):
        for e in self.envs:
            e.close()
