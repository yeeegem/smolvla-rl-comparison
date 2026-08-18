"""Critic head for Flow-SDE PPO.

The value function reads the same frozen conditioning the actor does: the
masked-mean-pooled VLM prefix output, concatenated with the normalized robot
state. Those features are already computed by
:meth:`grasprl.policy.smolvla_flow_sde.FlowSDESampler.encode`, so the critic
costs one small MLP forward and nothing else.

The value is predicted at the **outer** level, one number per policy decision
(one action chunk), matching the MDP the PPO trainer builds its GAE over.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ValueHead(nn.Module):
    def __init__(self, feature_dim: int, state_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # A near-zero output layer keeps the initial value estimates small, so
        # the first advantages are dominated by real returns rather than by an
        # arbitrary critic initialization.
        nn.init.zeros_(self.net[-1].bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, features: Tensor, state: Tensor) -> Tensor:
        x = torch.cat([features, state.to(features.dtype)], dim=-1)
        return self.net(x).squeeze(-1)


def build_value_head(sampler, hidden: int = 512) -> ValueHead:
    """Size a :class:`ValueHead` from the wrapped policy's dimensions.

    ``state_proj`` maps the padded state (``max_state_dim``) into the VLM's text
    hidden size, so its two shapes give both dimensions without hard-coding the
    SmolVLM variant.
    """
    proj = sampler.model.state_proj
    return ValueHead(feature_dim=proj.out_features, state_dim=proj.in_features,
                     hidden=hidden)
