"""Arm B of the comparison: Q-guided inference on a frozen SmolVLA policy.

Guided Action Flow (arXiv 2607.02092v1) never touches the policy weights. It
wraps SmolVLA's reverse-time flow sampler and, at every denoising step, nudges
the velocity along the gradient of a learned action-chunk critic. That is the
whole method, and it is the interesting counterpoint to Arm A: PPO buys its
improvement with 99.8M updated parameters and hours of on-policy rollout, while
this buys its improvement with a 4-layer MLP and no change to the policy at all.

Per step (paper Alg. lines 3-10)::

    v      = denoise_step(...).detach()            # base velocity, no autograd
    a_hat  = x_t - t * v                           # clean-action estimate  (3)
    Qbar   = mean_k Q_k(f_o, a_hat[..., :6], e_t)  #                        (6)
    g      = d Qbar / d a_hat                      #                        (7)
    m      = max(m_min, exp(-alpha * sigma_Q))     # disagreement gate      (8)
    v_g    = v - m * clip(g, c) / beta             #                        (9)
    x_next = x_t + dt * v_g

**The minus sign in (9) is the trap.** SmolVLA integrates from ``t = 1`` (noise)
down to ``t = 0`` (actions) along ``x_t = t*eps + (1-t)*a``, so the clean-action
estimate is ``a_hat = x_t - t*v``: *decreasing* v along g *increases* a_hat along
g, which is the direction that raises Q. A guidance rule copied from a
forward-time formulation would add here and walk the sampler downhill. The paper
flags this as the first of its three practical lessons; here it is pinned by
``tests/test_gaf.py::test_guidance_increases_the_critic_value``.

The base denoiser output is detached before the critic differentiates it, so the
critic gradient never flows back into the VLA -- guidance is a read-only
consumer of the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from grasprl.gaf.critic import CriticEnsemble, build_features


@dataclass
class GuidanceConfig:
    """Guidance hyperparameters. Defaults are the paper's best configuration.

    ``beta`` divides the gradient, so *smaller* beta is *stronger* guidance. The
    paper reports this as the sensitive knob: too strong and guidance overrules
    a base policy that was already competent, producing regressions on episodes
    that used to succeed.
    """

    beta: float = 2.0        # guidance strength divisor, Eq. (9)
    clip_norm: float = 1.0   # c, per-sample max gradient norm, Eq. (7)
    alpha: float = 10.0      # uncertainty scale in the gate, Eq. (8)
    min_gate: float = 0.1    # m_min, floor on the gate, Eq. (8)
    enabled: bool = True     # False = plain ODE, for a like-for-like baseline


class GuidedActionFlow:
    """Frozen SmolVLA + critic-gradient guidance at inference time."""

    def __init__(self, sampler, critic: CriticEnsemble, cfg: GuidanceConfig | None = None):
        self.sampler = sampler
        self.critic = critic.eval()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        self.cfg = cfg or GuidanceConfig()
        self.horizon = critic.cfg.horizon
        self.action_dim = critic.cfg.action_dim

    # -- the guided integration ---------------------------------------------

    def sample(self, prefix, state: Tensor, task_feature: Tensor | None = None,
               noise: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        """Integrate the guided flow and return the action chunk ``(B, chunk, 6)``.

        ``prefix`` is a :class:`grasprl.policy.smolvla_flow_sde.Prefix` and
        ``state`` the policy-preprocessed state, i.e. exactly the two things the
        critic was trained on.
        """
        s = self.sampler
        m = s.model
        ts, dt = s._timesteps(prefix.device)
        b = prefix.batch_size

        if noise is not None:
            x = noise
        elif generator is not None:
            x = torch.randn((b, s.chunk_size, s.max_action_dim), device=prefix.device,
                            dtype=torch.float32, generator=generator)
        else:
            x = m.sample_noise((b, s.chunk_size, s.max_action_dim), prefix.device)

        features = build_features(self.critic.cfg, state, prefix.pooled, task_feature)

        for i in range(s.num_steps):
            t = ts[i]
            with torch.no_grad():
                v = m.denoise_step(
                    prefix_pad_masks=prefix.pad_masks,
                    past_key_values=prefix.past_key_values,
                    x_t=x,
                    timestep=t.expand(b),
                ).to(dtype=torch.float32)

            if self.cfg.enabled:
                v = v + self._guidance(x, v, t, features)
            x = x + dt * v

        return x[:, :, : s.action_dim]

    def _guidance(self, x: Tensor, v: Tensor, t: Tensor, features: Tensor) -> Tensor:
        """The ``- m * clip(g, c) / beta`` term of Eq. (9), shaped like ``v``."""
        cfg = self.cfg
        with torch.enable_grad():
            a_hat = (x - t * v).detach().requires_grad_(True)      # Eq. (3)
            chunk = a_hat[:, : self.horizon, : self.action_dim]
            q_mean, q_std = self.critic.mean_and_std(features, chunk)   # Eq. (6)
            grad = torch.autograd.grad(q_mean.sum(), a_hat)[0]          # Eq. (7)

        grad = grad.detach()
        # Zero everything the critic did not see, so guidance can only move the
        # coordinates it actually has an opinion about: the executed horizon, and
        # only the physical action dimensions of SmolVLA's padded 32.
        mask = torch.zeros_like(grad)
        mask[:, : self.horizon, : self.action_dim] = 1.0
        grad = grad * mask

        # Per-sample norm clip, Eq. (7).
        flat = grad.flatten(start_dim=1)
        norm = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
        scale = (cfg.clip_norm / norm).clamp(max=1.0)
        grad = (flat * scale).view_as(grad)

        # Ensemble-disagreement gate, Eq. (8): back off where the members
        # disagree, which is the cheapest available proxy for "this chunk is
        # outside the critic's training distribution".
        gate = torch.clamp(torch.exp(-cfg.alpha * q_std), min=cfg.min_gate)
        gate = gate.view(-1, 1, 1)

        return -gate * grad / cfg.beta
