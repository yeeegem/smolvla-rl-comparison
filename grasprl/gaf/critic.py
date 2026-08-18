"""Action-chunk critic for Guided Action Flow (arXiv 2607.02092v1, Sec. III-E).

The critic scores a *candidate action chunk* in the context of what the policy is
currently looking at::

    Q_phi(f_o, a_{0:H-1}, e_tau) -> R                                    (Eq. 1)

and is trained by regression onto a sparse success-to-go target (Eq. 4-5). It is
never used to pick among a finite set of proposals -- Guided Action Flow needs
its *gradient with respect to the chunk*, so the critic must be differentiable in
the action and smooth enough that the gradient points somewhere useful.

Two design points differ from the paper, both deliberate:

* **Features.** The paper's critic reads a compact policy-side state and names
  that its main bottleneck ("richer visual-state features may improve value
  estimation"). Here the default ``state+pooled`` also feeds it the masked
  mean-pooled VLM prefix, which is free (``FlowSDESampler.encode`` already
  computes it) and is the *same* feature PPO's value head consumes -- so the two
  arms of the comparison have value functions of equal expressive power, which
  is a fairness requirement, not an optimisation. It also matters for this task
  specifically: whether a grasp is about to slip is visible in the wrist camera
  and essentially invisible in six joint angles. ``--critic-features state``
  reproduces the paper's compact variant.
* **Task feature.** ``e_tau`` is a single constant here (one task, one
  instruction), so it carries no information and is off by default. The plumbing
  is kept for parity with the paper's multi-family experiments.

The critic reads only the **physical** action dimensions. SmolVLA pads actions to
``max_action_dim = 32`` and only the first 6 mean anything; letting the critic
see the 26 padding coordinates would let it fit noise and, worse, would let
guidance push gradient into coordinates that do not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CriticConfig:
    """Architecture and training knobs. Defaults are the paper's."""

    hidden: int = 768
    depth: int = 4
    ensemble: int = 3          # K in Eq. (6); also supplies the uncertainty gate
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    gamma: float = 0.99        # discount of the success-to-go target, Eq. (4)
    features: str = "state+pooled"   # or "state"
    use_task_feature: bool = False
    action_dim: int = 6        # physical dims only; padding is excluded
    horizon: int = 10          # H, the chunk length the critic scores


class ActionChunkCritic(nn.Module):
    """One MLP member of the ensemble."""

    def __init__(self, feature_dim: int, cfg: CriticConfig):
        super().__init__()
        in_dim = feature_dim + cfg.horizon * cfg.action_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(d, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU()]
            d = cfg.hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.cfg = cfg

    def forward(self, features: Tensor, actions: Tensor) -> Tensor:
        """``features`` ``(B, F)``, ``actions`` ``(B, H, action_dim)`` -> ``(B,)``."""
        x = torch.cat([features, actions.flatten(start_dim=1)], dim=-1)
        return self.net(x).squeeze(-1)


class CriticEnsemble(nn.Module):
    """K independently initialised critics.

    The mean is the guidance signal (Eq. 6); the spread across members is the
    only uncertainty estimate available at every denoise step, and drives the
    disagreement gate (Eq. 8). It is a cheap heuristic, not a calibrated
    posterior -- the paper is explicit about that, and so is this.
    """

    def __init__(self, feature_dim: int, cfg: CriticConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim
        members = []
        for k in range(cfg.ensemble):
            torch.manual_seed(seed + 1000 * k)
            members.append(ActionChunkCritic(feature_dim, cfg))
        self.members = nn.ModuleList(members)

    def forward(self, features: Tensor, actions: Tensor) -> Tensor:
        """Stacked member values, ``(K, B)``."""
        return torch.stack([m(features, actions) for m in self.members])

    def mean_and_std(self, features: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """``(Qbar, sigma_Q)``, Eq. (6) and the input to Eq. (8).

        With a single member the std is zero, which makes the gate a no-op --
        the correct degenerate behaviour rather than a divide-by-zero.
        """
        q = self.forward(features, actions)
        if q.shape[0] == 1:
            return q[0], torch.zeros_like(q[0])
        return q.mean(0), q.std(0, unbiased=False)


def build_features(cfg: CriticConfig, state: Tensor, pooled: Tensor | None,
                   task_feature: Tensor | None) -> Tensor:
    """Assemble ``f_o`` (and ``e_tau``) into the critic's input vector.

    Kept as one function used by both training and inference so the two can
    never disagree about the layout -- a silent feature-order mismatch between
    collection and guidance would produce gradients that look plausible and are
    meaningless.
    """
    parts = [state]
    if cfg.features == "state+pooled":
        if pooled is None:
            raise ValueError("critic_features='state+pooled' needs the pooled VLM feature")
        parts.append(pooled)
    elif cfg.features != "state":
        raise ValueError(f"unknown critic features {cfg.features!r}")
    if cfg.use_task_feature:
        if task_feature is None:
            raise ValueError("use_task_feature=True needs a task feature")
        parts.append(task_feature)
    return torch.cat([p.to(parts[0].dtype) for p in parts], dim=-1)


def feature_dim(cfg: CriticConfig, state_dim: int, pooled_dim: int,
                task_dim: int = 0) -> int:
    d = state_dim
    if cfg.features == "state+pooled":
        d += pooled_dim
    if cfg.use_task_feature:
        d += task_dim
    return d
