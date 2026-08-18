"""Flow-SDE sampling for SmolVLA: turn a deterministic flow-matching policy into
a stochastic one with an exact, tractable log-probability, so PPO applies
unchanged.

Why this is needed
------------------
SmolVLA samples an action chunk by integrating a deterministic ODE. LeRobot's
``VLAFlowMatching.sample_actions`` runs ``K`` Euler steps from t=1 (pure noise)
down to t=0 (the action):

    x <- x + dt * v_theta(x, t),    dt = -1/K

A deterministic policy has no density to take a ratio of, so vanilla PPO has
nothing to clip. The fix used here is the standard ODE-to-SDE conversion: for
*any* noise schedule ``g(t)``, the stochastic process

    dx = [ v_theta(x, t) - (g(t)^2 / 2) * score(x, t) ] dt + g(t) dW,   dt < 0

has exactly the same time marginals ``p_t`` as the ODE. Substituting the drift
into the Fokker-Planck equation, the extra advection term and the new diffusion
term cancel identically, so nothing about the pretrained model's distribution is
distorted: at ``g = 0`` this reduces to the original sampler bit for bit, and at
``g > 0`` it is a different path to the same endpoint distribution. That gives
exploration for free, plus a Gaussian transition per denoise step whose
log-probability is exact.

**The sign of that correction depends on the direction of integration, and
getting it wrong is silent.** This chain runs from t=1 down to t=0, so ``dt`` is
negative and the correction *subtracts*; this is Anderson's reverse-time SDE. In
the forward (increasing t) convention the same term is written with a plus, and
copying that form into a backward loop makes the correction anti-cancel: the
sampled distribution is inflated instead of preserved, the policy silently gets
worse, and every test that only checks ``g = 0`` or the Gaussian arithmetic still
passes. ``test_sde_preserves_the_ode_marginals`` exists to pin this down.

The score in closed form
------------------------
SmolVLA trains on the linear path (see ``VLAFlowMatching.forward``):

    x_t = t * eps + (1 - t) * a,    target velocity u_t = eps - a

Eliminating ``a`` gives ``eps = (1 - t) * v + x``, and since
``p_t(x) = Integral N(x; (1-t) a, t^2 I) p(a) da`` the score is
``grad log p_t(x) = -E[eps | x] / t``. So

    score(x, t) = -( (1 - t) * v_theta(x, t) + x ) / t

No extra network and no score model: the velocity the policy already predicts is
all that is required.

Noise schedule
--------------
``g(t) = noise_scale * t``, which is proportional to the path's own noise level.
It vanishes as t -> 0, so the *final* denoise step adds almost no noise and the
executed action stays close to what the deterministic sampler would have
produced. The drift correction ``(g^2/2) * score`` also vanishes there, which
keeps the last step numerically well behaved (the ``1/t`` in the score is exactly
cancelled by the ``t^2`` in ``g^2``).

What is trainable
-----------------
The SigLIP vision encoder and the SmolLM2 language backbone are frozen and their
prefix forward pass runs under ``no_grad``. That single pass is the expensive one
(images through SigLIP, then the VLM, producing a KV cache). The ``K`` denoise
steps that follow only run the action expert, a narrower transformer over about
51 tokens cross-attending to that frozen cache, so backpropagating through the
whole chain is cheap and fits comfortably in 16 GB.

Which coordinates are scored
----------------------------
The latent chunk is ``(chunk_size=50, max_action_dim=32)``, but only the first 6
dims are the robot's action (the rest is padding the model is trained to drive to
zero) and only the first ``n_exec`` chunk steps are ever executed. Each denoise
transition is Gaussian with *diagonal* covariance, so the joint log-density
factorizes over coordinates given ``x_k``. Summing over only the executed,
non-padded coordinates therefore yields the exact conditional log-probability of
the block that actually reaches the environment, given the rest of the chain.
Because the stored chain is replayed verbatim during the PPO update, that
remainder is fixed (common random numbers), so the resulting ratio is a valid
importance weight and it carries none of the variance of the roughly 80 percent
of coordinates that cannot influence reward. Set ``score_all_coords`` to score
the full latent instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch import Tensor, nn

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class FlowSDEConfig:
    """Sampler settings. ``num_steps=None`` keeps the policy's own ``num_steps``."""

    num_steps: int | None = 4      # K during RL; evaluation uses the policy default (10)
    noise_scale: float = 0.6       # the constant in g(t) = noise_scale * t
    learn_noise: bool = False      # train a per-step multiplier on g(t)
    score_all_coords: bool = False # score the full latent instead of the executed block
    min_std: float = 1e-4          # floor on the per-step std, guards the last step
    # Train only the last N layers of the action expert. None trains all of them.
    # This is the compute escape hatch: dropping to a handful of layers, or to 0
    # (projections only), cuts optimizer state and backward cost roughly in
    # proportion, at the price of a less expressive update.
    expert_layers_trained: int | None = None


@dataclass
class Prefix:
    """Frozen per-observation conditioning: the VLM KV cache plus pooled features."""

    pad_masks: Tensor            # (B, L) prefix padding mask
    past_key_values: object      # DynamicCache produced by the frozen VLM
    pooled: Tensor               # (B, H) masked-mean-pooled prefix output, for the critic
    batch_size: int
    device: torch.device


@dataclass
class ChainOutput:
    """One sampled (or replayed) denoising chain."""

    chain: Tensor      # (B, K+1, chunk, dim): x_0 = noise ... x_K = the action latent
    log_prob: Tensor   # (B, K) per-denoise-step log-probability of the scored block
    entropy: Tensor    # (B, K) per-step Gaussian entropy of the scored block
    actions: Tensor    # (B, chunk, action_dim) unpadded action chunk, still normalized
    means: Tensor      # (B, K, chunk, dim) per-step transition means
    stds: Tensor       # (K,) per-step standard deviations, shared across coordinates


class FlowSDESampler(nn.Module):
    """Wraps a LeRobot ``SmolVLAPolicy`` with SDE sampling and exact log-probs.

    The wrapped policy is used, not modified: this calls its existing
    ``embed_prefix``, ``vlm_with_expert.forward`` and ``denoise_step``.
    """

    def __init__(self, policy, cfg: FlowSDEConfig | None = None, n_exec: int = 10):
        super().__init__()
        self.policy = policy
        self.cfg = cfg or FlowSDEConfig()
        self.n_exec = int(n_exec)
        self.action_dim = int(policy.config.action_feature.shape[0])
        self.chunk_size = int(policy.config.chunk_size)
        self.max_action_dim = int(policy.config.max_action_dim)
        # A per-denoise-step multiplier on g(t), in log space so it stays positive.
        # Fixed at 1 unless learn_noise is set, in which case PPO can widen or
        # narrow exploration the way a learned log-std would.
        self.log_noise_gain = nn.Parameter(
            torch.zeros(self.num_steps), requires_grad=bool(self.cfg.learn_noise)
        )

    @property
    def num_steps(self) -> int:
        return int(self.cfg.num_steps or self.policy.config.num_steps)

    @property
    def model(self):
        """The underlying ``VLAFlowMatching`` module."""
        return self.policy.model

    # -- conditioning --------------------------------------------------------

    @torch.no_grad()
    def encode(self, batch: dict[str, Tensor]) -> Prefix:
        """Run the frozen vision and language stack once and cache its output.

        ``batch`` must already have been through the policy's preprocessor.
        """
        m = self.model
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        embs, pad_masks, att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        outputs, kv = m.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=m.config.use_cache,
        )
        # Masked mean over real prefix tokens: a fixed-width summary of what the
        # policy is looking at, which the critic reads.
        h = outputs[0].to(dtype=torch.float32)
        mask = pad_masks.to(h.dtype).unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return Prefix(pad_masks=pad_masks, past_key_values=kv, pooled=pooled,
                      batch_size=state.shape[0], device=state.device)

    # -- SDE integration -----------------------------------------------------

    def _timesteps(self, device) -> tuple[Tensor, float]:
        k = self.num_steps
        dt = -1.0 / k
        # t = 1, 1 - 1/K, ..., 1/K, matching VLAFlowMatching.sample_actions.
        t = torch.tensor([1.0 + i * dt for i in range(k)], dtype=torch.float32, device=device)
        return t, dt

    def _step_std(self, t: Tensor, dt: float, index: int) -> Tensor:
        """Standard deviation of one Euler-Maruyama step: g(t) * sqrt(|dt|)."""
        g = self.cfg.noise_scale * t * torch.exp(self.log_noise_gain[index])
        return (g * math.sqrt(abs(dt))).clamp(min=self.cfg.min_std)

    def scored_block(self, x: Tensor) -> Tensor:
        """Slice the coordinates whose log-probability enters the PPO ratio."""
        if self.cfg.score_all_coords:
            return x
        return x[:, : self.n_exec, : self.action_dim]

    @property
    def n_scored(self) -> int:
        """Number of scored coordinates per denoise step.

        Log-probabilities and KLs are sums over these, so dividing by this is what
        turns them into per-coordinate quantities with the magnitudes PPO
        conventions (``target_kl`` around 0.01) were written for.
        """
        if self.cfg.score_all_coords:
            return self.chunk_size * self.max_action_dim
        return self.n_exec * self.action_dim

    def rollout(
        self,
        prefix: Prefix,
        chain: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> ChainOutput:
        """Sample a denoising chain, or replay a stored one to re-score it.

        Passing ``chain`` makes this deterministic in the noise: the same states
        are visited and only the means change with the current parameters, which
        is exactly what a PPO importance ratio needs.

        Passing ``noise`` fixes only the starting point ``x_0``, which is what
        lets ``scripts/measure_noise_scale.py`` compare an SDE sample against the
        ODE sample grown from the same seed.
        """
        m = self.model
        b, device = prefix.batch_size, prefix.device
        ts, dt = self._timesteps(device)
        k = self.num_steps
        shape = (b, self.chunk_size, self.max_action_dim)

        if chain is None:
            if noise is not None:
                x = noise
            elif generator is None:
                x = m.sample_noise(shape, device)
            else:
                x = torch.randn(shape, device=device, generator=generator,
                                dtype=torch.float32)
            states = [x]
        else:
            if chain.shape[1] != k + 1:
                raise ValueError(
                    f"stored chain has {chain.shape[1] - 1} denoise steps but the "
                    f"sampler is configured for {k}; they must match"
                )
            states = list(chain.unbind(dim=1))
            x = states[0]

        log_probs, entropies, means, stds = [], [], [], []
        for i in range(k):
            t = ts[i]
            t_batch = t.expand(b)
            v = m.denoise_step(
                prefix_pad_masks=prefix.pad_masks,
                past_key_values=prefix.past_key_values,
                x_t=x,
                timestep=t_batch,
            ).to(dtype=torch.float32)

            # score(x, t) = -((1 - t) v + x) / t, exact for the linear path.
            # The correction SUBTRACTS because the chain runs backwards in time
            # (dt < 0). See the module docstring; the sign is pinned by
            # tests/test_flow_sde.py::test_sde_preserves_the_ode_marginals.
            score = -((1.0 - t) * v + x) / t
            mean = x + dt * (v - 0.5 * (self.cfg.noise_scale * t) ** 2 * score)
            std = self._step_std(t, dt, i)

            if chain is None:
                eps = torch.randn(mean.shape, device=device, dtype=mean.dtype,
                                  generator=generator)
                x_next = mean + std * eps
                states.append(x_next)
            else:
                x_next = states[i + 1]

            log_probs.append(_gaussian_log_prob(self.scored_block(x_next), self.scored_block(mean), std))
            entropies.append(_gaussian_entropy(self.scored_block(mean), std))
            means.append(mean)
            stds.append(std)
            x = x_next

        latent = states[-1]
        actions = latent[:, :, : self.action_dim]
        if self.policy.config.adapt_to_pi_aloha:
            actions = self.policy._pi_aloha_encode_actions(actions)
        return ChainOutput(
            chain=torch.stack(states, dim=1),
            log_prob=torch.stack(log_probs, dim=1),
            entropy=torch.stack(entropies, dim=1),
            actions=actions,
            means=torch.stack(means, dim=1),
            stds=torch.stack(stds),
        )

    @torch.no_grad()
    def sample_ode(self, prefix: Prefix, noise: Tensor | None = None,
                   generator: torch.Generator | None = None) -> Tensor:
        """Deterministic sampling, identical to LeRobot's ``sample_actions``.

        Used for evaluation, and, by passing the same ``noise``, as the reference
        that ``tests/test_flow_sde.py`` checks the SDE machinery against.
        """
        m = self.model
        ts, dt = self._timesteps(prefix.device)
        shape = (prefix.batch_size, self.chunk_size, self.max_action_dim)
        if noise is not None:
            x = noise
        elif generator is not None:
            # Seeded explicitly: an unseeded evaluator was worth ~0.10 of
            # run-to-run success variation in the source repo, which is larger
            # than most effects this comparison tries to measure.
            x = torch.randn(shape, device=prefix.device, dtype=torch.float32,
                            generator=generator)
        else:
            x = m.sample_noise(shape, prefix.device)
        for i in range(self.num_steps):
            v = m.denoise_step(
                prefix_pad_masks=prefix.pad_masks,
                past_key_values=prefix.past_key_values,
                x_t=x,
                timestep=ts[i].expand(prefix.batch_size),
            )
            x = x + dt * v
        return x[:, :, : self.action_dim]

    # -- parameter groups ----------------------------------------------------

    def expert_layers(self) -> nn.ModuleList:
        """The action expert's transformer layers, in order."""
        expert = getattr(self.model.vlm_with_expert, "lm_expert", None)
        if expert is None:
            raise AttributeError(
                "SmolVLMWithExpertModel has no `lm_expert`; the action expert must "
                "be locatable for PPO to train it"
            )
        layers = getattr(expert, "layers", None)
        if layers is None:
            layers = expert.model.layers
        return layers

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Everything PPO updates inside the policy: the action expert (or its
        last ``expert_layers_trained`` layers) and the projections that feed and
        read it. The VLM stays frozen."""
        m = self.model
        mods: list[nn.Module] = [
            m.action_in_proj, m.action_out_proj,
            m.action_time_mlp_in, m.action_time_mlp_out,
        ]
        layers = self.expert_layers()
        n = self.cfg.expert_layers_trained
        mods += list(layers) if n is None else list(layers)[len(layers) - n:] if n > 0 else []
        params = [p for mod in mods for p in mod.parameters()]
        if self.cfg.learn_noise:
            params.append(self.log_noise_gain)
        return params

    def freeze_backbone(self) -> None:
        """Freeze everything, then re-enable only the trainable set."""
        for p in self.policy.parameters():
            p.requires_grad_(False)
        for p in self.trainable_parameters():
            p.requires_grad_(True)


def _gaussian_log_prob(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Sum of an isotropic Gaussian log-density over every non-batch dim."""
    var = std**2
    lp = -0.5 * (((x - mean) ** 2) / var + LOG_2PI) - torch.log(std)
    return lp.flatten(start_dim=1).sum(dim=1)


def _gaussian_entropy(mean: Tensor, std: Tensor) -> Tensor:
    """Entropy of the same Gaussian, summed over the scored coordinates.

    It depends only on ``std``, so with ``learn_noise = False`` this is a constant
    and an entropy bonus has no gradient. It is still logged, because a shrinking
    entropy is the signal that a learned noise schedule is collapsing.
    """
    n = mean[0].numel()
    return (n * (0.5 * (LOG_2PI + 1.0) + torch.log(std))).expand(mean.shape[0])


def chain_kl(mean_new: Tensor, mean_ref: Tensor, std: Tensor) -> Tensor:
    """KL between two denoise steps that share a std: ``||dmean||^2 / (2 std^2)``.

    Used for the proximal penalty against the frozen imitation policy. Pass the
    means already sliced by :meth:`FlowSDESampler.scored_block` so this is
    measured on the same coordinates as the PPO ratio, and divide the result by
    ``n_scored`` to read it per coordinate.
    """
    d = (mean_new - mean_ref) ** 2
    return (d.flatten(start_dim=1).sum(dim=1)) / (2.0 * std**2)
