"""Guided Action Flow: the pieces that are silent when wrong.

Three of these tests exist because the failure mode is a plausible-looking
number rather than an exception:

* the guidance sign -- a copied forward-time rule walks the sampler *downhill*
  and simply reports a worse policy;
* the padding mask -- a critic that reads SmolVLA's 26 unused action dimensions
  fits noise and pushes gradient into coordinates that do not exist;
* the train/val split -- splitting by chunk instead of by episode leaks
  trajectory context and reports a critic far better than it is.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grasprl.gaf.critic import CriticConfig, CriticEnsemble, build_features, feature_dim
from grasprl.gaf.train_critic import split_by_episode, success_to_go

# ---------------------------------------------------------------------------
# Targets and splits
# ---------------------------------------------------------------------------

def test_success_to_go_decays_backwards_from_the_first_success():
    """Eq. (4): y_i = gamma^(j*-i), 0 for an episode that never succeeds."""
    success = np.array([0, 0, 0, 1, 0,      # episode 0 succeeds at index 3
                        0, 0, 0], np.int8)  # episode 1 never does
    episode = np.array([0, 0, 0, 0, 0, 1, 1, 1], np.int32)
    y = success_to_go(success, episode, gamma=0.9)
    assert y[:5] == pytest.approx([0.9**3, 0.9**2, 0.9, 1.0, 1.0])
    assert y[5:] == pytest.approx([0.0, 0.0, 0.0])


def test_split_is_episode_level():
    """No episode may appear on both sides, or the critic grades its own homework."""
    episode = np.repeat(np.arange(20), 7)
    tr, va = split_by_episode(episode, val_fraction=0.25, seed=0)
    assert len(tr) and len(va)
    assert not (set(episode[tr]) & set(episode[va]))
    assert len(tr) + len(va) == len(episode)


# ---------------------------------------------------------------------------
# Critic plumbing
# ---------------------------------------------------------------------------

def _cfg(**kw):
    return CriticConfig(**{"hidden": 32, "depth": 3, "ensemble": 3,
                           "horizon": 4, "action_dim": 6, **kw})


def test_feature_layout_matches_declared_dim():
    cfg = _cfg()
    state = torch.zeros(5, 32)
    pooled = torch.zeros(5, 960)
    f = build_features(cfg, state, pooled, None)
    assert f.shape[1] == feature_dim(cfg, 32, 960)

    compact = _cfg(features="state")
    assert build_features(compact, state, None, None).shape[1] == feature_dim(compact, 32, 960)


def test_critic_reads_only_the_physical_action_dims():
    """SmolVLA pads actions to 32; the critic's input width must ignore the rest."""
    cfg = _cfg()
    ens = CriticEnsemble(feature_dim(cfg, 32, 960), cfg)
    first = ens.members[0].net[0]
    assert first.in_features == feature_dim(cfg, 32, 960) + cfg.horizon * cfg.action_dim


def test_ensemble_reports_mean_and_disagreement():
    cfg = _cfg()
    fd = feature_dim(cfg, 32, 960)
    ens = CriticEnsemble(fd, cfg, seed=0)
    f, a = torch.randn(7, fd), torch.randn(7, cfg.horizon, cfg.action_dim)
    q = ens(f, a)
    mean, std = ens.mean_and_std(f, a)
    assert q.shape == (cfg.ensemble, 7)
    assert torch.allclose(mean, q.mean(0))
    # Independently initialised members must actually disagree, or the gate is dead.
    assert float(std.mean()) > 0


# ---------------------------------------------------------------------------
# The guided update
# ---------------------------------------------------------------------------

class _QuadraticCritic(torch.nn.Module):
    """Stand-in critic whose maximum is a known point, so guidance is checkable."""

    class _Cfg:
        horizon = 4
        action_dim = 6
        features = "state"
        use_task_feature = False

    def __init__(self, target: torch.Tensor, spread: float = 0.0):
        super().__init__()
        self.cfg = self._Cfg()
        self.target = target
        self.spread = spread
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def parameters(self, recurse=True):
        return iter([self._dummy])

    def mean_and_std(self, features, actions):
        q = -((actions - self.target) ** 2).flatten(start_dim=1).sum(-1)
        return q, torch.full_like(q, self.spread)


class _StubSampler:
    """The two attributes GuidedActionFlow needs from a real sampler."""

    chunk_size = 8
    max_action_dim = 32
    action_dim = 6
    num_steps = 1

    def _timesteps(self, device):
        return torch.tensor([0.5], device=device), -1.0


def _guidance_delta(target, x, v, beta=2.0, alpha=10.0, spread=0.0, clip=1e9, min_gate=0.0):
    from grasprl.gaf.guided_sampler import GuidanceConfig, GuidedActionFlow

    g = GuidedActionFlow(_StubSampler(), _QuadraticCritic(target, spread),
                         GuidanceConfig(beta=beta, clip_norm=clip, alpha=alpha,
                                        min_gate=min_gate))
    t = torch.tensor(0.5)
    features = torch.zeros(1, 1)
    return g._guidance(x, v, t, features)


def test_guidance_increases_the_critic_value():
    """The whole method in one assertion, and the sign trap it guards.

    SmolVLA integrates t: 1 -> 0 with a_hat = x - t*v, so *decreasing* v along
    the critic gradient *increases* a_hat along it. A rule copied from a
    forward-time formulation adds instead, and quietly steers downhill.
    """
    torch.manual_seed(0)
    x = torch.zeros(1, 8, 32)
    v = torch.randn(1, 8, 32) * 0.1
    target = torch.ones(1, 4, 6) * 0.7
    t = 0.5

    delta = _guidance_delta(target, x, v)
    q = lambda a: -((a[:, :4, :6] - target) ** 2).sum()
    a_before = x - t * v
    a_after = x - t * (v + delta)
    assert q(a_after) > q(a_before), "guidance must move the chunk toward higher Q"


def test_guidance_touches_only_the_scored_coordinates():
    """Padding dims and un-executed chunk steps must be left exactly alone."""
    torch.manual_seed(0)
    delta = _guidance_delta(torch.ones(1, 4, 6) * 0.7,
                            torch.zeros(1, 8, 32), torch.randn(1, 8, 32) * 0.1)
    assert delta[:, :4, :6].abs().sum() > 0
    assert delta[:, 4:, :].abs().sum() == 0     # beyond the critic's horizon
    assert delta[:, :, 6:].abs().sum() == 0     # SmolVLA's padding dimensions


def test_disagreement_gate_backs_off_when_the_ensemble_is_unsure():
    """Eq. (8): guidance shrinks as sigma_Q grows, but never below m_min."""
    torch.manual_seed(0)
    x, v = torch.zeros(1, 8, 32), torch.randn(1, 8, 32) * 0.1
    target = torch.ones(1, 4, 6) * 0.7
    confident = _guidance_delta(target, x, v, spread=0.0, min_gate=0.1).norm()
    unsure = _guidance_delta(target, x, v, spread=1.0, min_gate=0.1).norm()
    assert unsure < confident
    # The floor keeps a little guidance alive rather than switching it off.
    assert float(unsure) == pytest.approx(float(confident) * 0.1, rel=1e-4)


def test_beta_scales_guidance_inversely():
    """beta DIVIDES the gradient: larger beta must mean gentler guidance."""
    torch.manual_seed(0)
    x, v = torch.zeros(1, 8, 32), torch.randn(1, 8, 32) * 0.1
    target = torch.ones(1, 4, 6) * 0.7
    strong = _guidance_delta(target, x, v, beta=1.0).norm()
    weak = _guidance_delta(target, x, v, beta=4.0).norm()
    assert float(strong) == pytest.approx(4 * float(weak), rel=1e-4)


def test_gradient_is_norm_clipped():
    """Eq. (7)'s clip caps the per-sample step regardless of critic scale."""
    torch.manual_seed(0)
    x, v = torch.zeros(1, 8, 32), torch.randn(1, 8, 32) * 0.1
    huge = torch.ones(1, 4, 6) * 500.0     # enormous gradient
    delta = _guidance_delta(huge, x, v, beta=1.0, clip=1.0, min_gate=1.0, alpha=0.0)
    assert float(delta.norm()) == pytest.approx(1.0, rel=1e-4)
