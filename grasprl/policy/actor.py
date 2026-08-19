"""One interface for the three things being compared.

``base``, ``ppo`` and ``gaf`` differ in *where* the improvement lives -- PPO in
the weights, GAF in an inference-time critic, base nowhere -- but a fair
comparison requires that everything downstream treats them identically. So all
three are wrapped as an :class:`Actor` that maps a list of observations to an
executable action chunk, and the sim evaluator, the calibration sweep and the
real-arm harness all drive that one interface.

Both arms sample with the **deterministic ODE** at the checkpoint's native flow
steps. PPO trains against a noisy SDE sampler, but that noise is an exploration
device with a real cost -- in the source repo the best SDE setting scored 0.30
against the ODE's 0.45 -- so evaluating PPO with it would score the exploration
mechanism rather than the learned policy.

The sampler noise is seeded explicitly. Leaving it to the global RNG was worth
about 0.10 of run-to-run variation in the source repo's evaluator, which is
larger than most of the effects this comparison is trying to measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from grasprl.policy.loader import load_smolvla, observations_to_batch, postprocess_actions
from grasprl.policy.smolvla_flow_sde import FlowSDEConfig, FlowSDESampler

METHODS = ("base", "ppo", "gaf")


class Actor:
    """Frozen or fine-tuned SmolVLA, optionally under critic guidance."""

    def __init__(self, checkpoint: str | Path, device: torch.device, task: str,
                 n_exec: int = 10, critic_dir: str | Path | None = None,
                 guidance=None, seed: int = 0, num_steps: int | None = None,
                 stochastic: bool = False, noise_scale: float = 0.2):
        self.device = device
        self.task = task
        self.n_exec = n_exec
        # `stochastic` scores the sampler PPO actually explores with, rather than
        # the deterministic one it is evaluated with. The gap between the two is
        # the exploration tax, and it has to be measured rather than assumed: at
        # noise_scale 0.2 / K=4 it turned out to cost the entire task (0% success
        # against the ODE's 10%), which silently starved PPO of any positive
        # reward and taught it to stop grasping.
        self.stochastic = stochastic
        self.policy, self.pre, self.post = load_smolvla(checkpoint, device)
        self.sampler = FlowSDESampler(
            self.policy,
            FlowSDEConfig(num_steps=num_steps, noise_scale=noise_scale),
            n_exec=n_exec).to(device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(500_000 + 7919 * seed)

        self.guided = None
        if critic_dir is not None:
            from grasprl.gaf.guided_sampler import GuidanceConfig, GuidedActionFlow
            from grasprl.gaf.train_critic import load_critic

            critic = load_critic(critic_dir, device)
            self.guided = GuidedActionFlow(self.sampler, critic,
                                           guidance or GuidanceConfig())

    @property
    def method(self) -> str:
        return "gaf" if self.guided is not None else "policy"

    @torch.no_grad()
    def act(self, observations: list[dict]) -> np.ndarray:
        """``(B, n_exec, 6)`` actions in LeRobot calibrated units."""
        batch = observations_to_batch(observations, self.task, self.pre, self.device)
        prefix = self.sampler.encode(batch)
        if self.guided is None and self.stochastic:
            latent = self.sampler.rollout(prefix, generator=self.generator).actions
        elif self.guided is None:
            latent = self.sampler.sample_ode(prefix, generator=self.generator)
        else:
            state = self.policy.prepare_state(batch)
            latent = self.guided.sample(prefix, state, generator=self.generator)
        return postprocess_actions(latent[:, : self.n_exec], self.post)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.generator.manual_seed(500_000 + 7919 * seed)


def build_actor(method: str, checkpoint: str | Path, device: torch.device, task: str,
                n_exec: int = 10, critic_dir: str | Path | None = None,
                guidance=None, seed: int = 0, stochastic: bool = False,
                noise_scale: float = 0.2, num_steps: int | None = None) -> Actor:
    """Construct the actor for one arm of the comparison.

    ``gaf`` deliberately takes the **base** checkpoint: the whole claim of the
    method is that the policy is untouched, so pointing it at a PPO checkpoint
    would silently compare a different thing.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "gaf" and critic_dir is None:
        raise ValueError("method='gaf' needs --critic pointing at a trained critic")
    if method != "gaf" and critic_dir is not None:
        raise ValueError(f"method={method!r} does not use a critic; drop --critic")
    return Actor(checkpoint, device, task, n_exec=n_exec,
                 critic_dir=critic_dir if method == "gaf" else None,
                 guidance=guidance, seed=seed, stochastic=stochastic,
                 noise_scale=noise_scale, num_steps=num_steps)
