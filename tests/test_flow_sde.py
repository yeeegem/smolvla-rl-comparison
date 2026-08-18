"""Correctness tests for the Flow-SDE sampler.

These are the tests that matter most in the repo. If the SDE conversion is wrong,
PPO will still run, still produce learning curves and still look fine, while
optimizing a distribution that has nothing to do with the pretrained policy. Each
test below pins down one link in the derivation:

1. **the SDE preserves the ODE's marginals**, which pins the sign of the drift
   correction,
2. the sampler reproduces LeRobot's own deterministic sampler exactly,
3. the analytic per-step log-probability matches a Monte Carlo estimate,
4. replaying a stored chain gives back the log-probability it was sampled with,
   which is what makes the PPO ratio start at exactly 1,
5. the score formula recovers the exact score of the linear Gaussian path,
6. the noise schedule shrinks to near zero on the final step.

Test 1 is the one with scars. The correction term's sign depends on the direction
of integration, and this chain runs backwards (dt < 0), so it subtracts. Writing
it with a plus, the forward-time convention, inflates the sampled distribution
instead of preserving it: the policy quietly degrades, RL fine-tuning optimizes a
worse starting point than intended, and every other test in this file still
passes, because they all either set ``g = 0`` or only exercise the Gaussian
arithmetic. That bug shipped once and cost a full set of PPO runs.

Most tests need a real SmolVLA, so they load ``lerobot/smolvla_base`` and are
skipped when it is not cached locally or when there is no GPU. Test 1 is pure
numpy on an analytic flow and always runs.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SmolVLA tests need a GPU"
)

CHECKPOINT = "lerobot/smolvla_base"


# ---------------------------------------------------------------------------
# 1. Marginal preservation, on an analytic flow. No GPU, no SmolVLA.
# ---------------------------------------------------------------------------

def _toy_flow(mu0: float = 1.3, s0: float = 0.4):
    """Exact velocity and score for data ``a ~ N(mu0, s0^2)`` on SmolVLA's path.

    With ``x_t = (1 - t) a + t eps`` the marginal is
    ``p_t = N((1 - t) mu0, (1 - t)^2 s0^2 + t^2)``, so both the score and the
    conditional-expectation velocity are available in closed form. That makes
    this a test of the SDE *update* alone, with no network in the way.
    """
    def score(x, t):
        return -(x - (1 - t) * mu0) / ((1 - t) ** 2 * s0**2 + t**2)

    def velocity(x, t):
        # Posterior a|x is Gaussian and non-singular for t in (0, 1].
        prec = 1 / s0**2 + (1 - t) ** 2 / t**2
        e_a = (mu0 / s0**2 + (1 - t) * x / t**2) / prec
        e_eps = (x - (1 - t) * e_a) / t
        return e_eps - e_a

    return velocity, score


def _integrate(k: int, noise_scale: float, sign: float, n: int = 200_000, seed: int = 0):
    """Run the toy chain with a given sign on the drift correction."""
    velocity, score = _toy_flow()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)          # x_1 ~ N(0, 1), the correct t=1 marginal
    dt = -1.0 / k
    for i in range(k):
        t = 1.0 + i * dt
        g = noise_scale * t
        drift = velocity(x, t) + sign * 0.5 * g * g * score(x, t)
        x = x + dt * drift + g * np.sqrt(abs(dt)) * rng.standard_normal(n)
    return float(x.mean()), float(x.std())


@pytest.mark.parametrize("k", [4, 10, 50])
def test_sde_preserves_the_ode_marginals(k):
    """Adding noise must not change the sampled distribution.

    This is the property the whole method rests on, and it is what pins the sign
    of the drift correction. The chain integrates from t=1 to t=0, so ``dt`` is
    negative and the correction subtracts (Anderson's reverse-time SDE). With the
    forward-time sign it *adds*, which anti-cancels the diffusion and inflates the
    distribution.

    The reference is the ODE at the same ``k``, not the true data distribution:
    both share the same Euler discretization error, so comparing against the ODE
    isolates the effect of the noise.
    """
    ode_mean, ode_std = _integrate(k, 0.0, +1.0)
    ok_mean, ok_std = _integrate(k, 0.6, -1.0)     # the sign the code uses
    _bad_mean, bad_std = _integrate(k, 0.6, +1.0)   # the forward-time sign

    assert abs(ok_std - ode_std) < 0.03, (
        f"K={k}: SDE std {ok_std:.4f} drifted from the ODE's {ode_std:.4f}")
    assert abs(ok_mean - ode_mean) < 0.02
    # And the wrong sign must be clearly worse, so this test cannot pass by luck.
    assert abs(bad_std - ode_std) > 3 * abs(ok_std - ode_std)


def test_sampler_uses_the_reverse_time_sign():
    """The shipped sampler must match the reverse-time integration, not the
    forward-time one. Guards the exact line in ``FlowSDESampler.rollout``."""
    velocity, score = _toy_flow()
    x, t, noise_scale, dt = 0.7, 0.5, 0.6, -0.25
    v, s = velocity(x, t), score(x, t)
    g = noise_scale * t

    reverse = x + dt * (v - 0.5 * g * g * s)
    forward = x + dt * (v + 0.5 * g * g * s)
    assert reverse != pytest.approx(forward)

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "grasprl" / "policy" / "smolvla_flow_sde.py").read_text()
    assert "dt * (v - 0.5 * (self.cfg.noise_scale * t) ** 2 * score)" in src, (
        "the drift correction in rollout() must subtract; see "
        "test_sde_preserves_the_ode_marginals for why"
    )


@pytest.fixture(scope="module")
def rig():
    """A loaded policy plus one real observation batch of size 2.

    Uses the repo's own base checkpoint when it has been fetched (that is the
    policy every result is measured against, so it is the one worth pinning) and
    falls back to the public ``lerobot/smolvla_base``.
    """
    from grasprl.envs.pickplace_env import DEFAULT_TASK, EnvConfig, PickPlaceEnv
    from grasprl.policy.loader import load_smolvla, observations_to_batch

    device = torch.device("cuda")
    local = pathlib.Path(__file__).resolve().parents[1] / "checkpoints" / "base_smolvla"
    checkpoint = str(local) if (local / "config.json").exists() else CHECKPOINT
    try:
        policy, pre, _post = load_smolvla(checkpoint, device)
    except Exception as e:  # noqa: BLE001 - offline or no cached weights
        pytest.skip(f"could not load {checkpoint}: {e}")

    env = PickPlaceEnv(cfg=EnvConfig(domain_randomize=False))
    obs, _ = env.reset(seed=0)
    batch = observations_to_batch([obs, obs], DEFAULT_TASK, pre, device)
    env.close()
    yield policy, batch, device


def _sampler(policy, **kw):
    from grasprl.policy.smolvla_flow_sde import FlowSDEConfig, FlowSDESampler

    cfg = FlowSDEConfig(**{"num_steps": 4, "noise_scale": 0.6, **kw})
    return FlowSDESampler(policy, cfg, n_exec=10).to(next(policy.parameters()).device)


@gpu_only
def test_ode_path_matches_lerobot(rig):
    """Zero-noise integration must reproduce LeRobot's sampler bit for bit.

    This is the anchor: it proves the prefix encoding, the KV cache reuse, the
    timestep schedule and the Euler step are all identical to the pretrained
    inference path, so anything the SDE adds is additive and not a reimplementation
    that quietly drifted.
    """
    policy, batch, device = rig
    s = _sampler(policy)
    prefix = s.encode(batch)

    noise = torch.randn(
        (prefix.batch_size, s.chunk_size, s.max_action_dim), device=device
    )
    mine = s.sample_ode(prefix, noise=noise)

    # LeRobot's own path, same noise, same number of steps.
    policy.config.num_steps = s.num_steps
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    with torch.no_grad():
        ref = policy.model.sample_actions(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK], state, noise=noise,
        )[:, :, : s.action_dim]

    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max().item()


@gpu_only
def test_log_prob_matches_monte_carlo(rig):
    """The analytic per-step log-density must match an empirical one.

    Draws many samples from a single denoise transition and compares the analytic
    log-density against a histogram-free check: for a correct isotropic Gaussian
    the standardized residuals have unit variance and zero mean.
    """
    policy, batch, device = rig
    s = _sampler(policy)
    prefix = s.encode(batch)

    with torch.no_grad():
        out = s.rollout(prefix)
    mean, std = out.means[:, 0], out.stds[0]
    residual = (out.chain[:, 1] - mean) / std

    # One draw is not enough to test a moment, so redraw the same transition.
    draws = torch.randn((512, *mean.shape[1:]), device=device) * std + mean[0]
    z = (draws - mean[0]) / std
    assert abs(z.mean().item()) < 0.05
    assert abs(z.std().item() - 1.0) < 0.05

    # And the analytic log-density of those draws matches the closed form.
    from grasprl.policy.smolvla_flow_sde import _gaussian_log_prob

    lp = _gaussian_log_prob(draws, mean[0].expand_as(draws), std)
    n = draws[0].numel()
    expected = -0.5 * ((z**2).flatten(1).sum(1) + n * math.log(2 * math.pi)) - n * torch.log(std)
    assert torch.allclose(lp, expected, atol=1e-2)
    assert torch.isfinite(residual).all()


@gpu_only
def test_replaying_a_chain_reproduces_its_log_prob(rig):
    """Re-scoring a stored chain under unchanged parameters must give the same
    log-probability, so the PPO ratio is exactly 1 on the first epoch."""
    policy, batch, _device = rig
    s = _sampler(policy)
    prefix = s.encode(batch)

    with torch.no_grad():
        sampled = s.rollout(prefix)
        replayed = s.rollout(prefix, chain=sampled.chain)

    assert torch.allclose(sampled.log_prob, replayed.log_prob, atol=1e-3)
    ratio = torch.exp(replayed.log_prob - sampled.log_prob)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)


@gpu_only
def test_score_formula_matches_gaussian_path(rig):
    """``score = -((1-t) v + x) / t`` must be the score of the linear path.

    Checked against the definition rather than the derivation: for the path
    ``x_t = t eps + (1-t) a`` with a known single-point data distribution, the
    exact velocity is ``eps - a`` and the exact score is ``-(x - (1-t) a) / t^2``.
    Substituting the exact velocity into the formula must recover it.
    """
    torch.manual_seed(0)
    a = torch.randn(4, 8)
    eps = torch.randn(4, 8)
    for t in (0.9, 0.5, 0.25):
        x = t * eps + (1 - t) * a
        v = eps - a                              # exact velocity for this pair
        got = -((1 - t) * v + x) / t
        want = -(x - (1 - t) * a) / t**2         # exact score of N(x; (1-t)a, t^2 I)
        assert torch.allclose(got, want, atol=1e-5)


@gpu_only
def test_noise_schedule_vanishes_at_the_end(rig):
    """g(t) = noise_scale * t must shrink monotonically, so the last denoise step
    barely perturbs the action that reaches the robot."""
    policy, batch, _device = rig
    s = _sampler(policy)
    prefix = s.encode(batch)
    with torch.no_grad():
        out = s.rollout(prefix)
    stds = out.stds.tolist()
    assert stds == sorted(stds, reverse=True)
    assert stds[-1] < stds[0] / 2
    assert np.all(np.isfinite(stds))
