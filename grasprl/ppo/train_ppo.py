"""Arm A of the comparison: PPO on SmolVLA's flow-matching sampler.

SmolVLA generates an action chunk by integrating a *deterministic* flow ODE, so
it has no action density and nothing for PPO to clip. The fix, ported from
an earlier unpublished project, is to replace that ODE with the marginal-
preserving reverse-time SDE (see ``grasprl/policy/smolvla_flow_sde.py``): the
sampler becomes a chain of Gaussian transitions with an exact log-density, while
still drawing from the same distribution the imitation policy learned.

The MDP has two levels:

* **outer** -- one policy decision, i.e. one chunk of which ``n_exec`` actions
  are executed. Reward, discount, GAE and the value head all live here, so the
  MDP's timestep is the policy's actual decision point.
* **inner** -- the K denoise transitions inside one decision. No reward, no
  discount; every inner step inherits its decision's advantage. The PPO ratio is
  formed **per inner step**, which gives K times finer clipping than treating a
  whole chunk as one action.

Unlike Arm B (Guided Action Flow), this method **changes the policy weights**:
the action expert and the four action projections train (99.8M of 450M), while
SigLIP and SmolLM2 stay frozen.

Three traps are carried over from the source repo with their tests, because each
one is silent -- the losses look healthy while the policy quietly gets worse:

1. **The drift correction subtracts.** The chain runs backwards in time, so the
   Anderson reverse-time SDE flips the sign relative to the forward-time
   formula. Getting this wrong cost a factor of 7 in success rate there. Pinned
   by ``tests/test_flow_sde.py::test_sde_preserves_the_ode_marginals`` and
   ``::test_sampler_uses_the_reverse_time_sign``.
2. **KL units.** One decision is an ``n_exec x 6`` block pushed through K steps,
   so a raw log-density is a sum over 60 coordinates and summed KLs run about
   two orders of magnitude above the values PPO conventions assume. Both KLs are
   divided by ``n_scored`` before being logged or thresholded. The *ratio* is
   never normalised -- it has to stay the true density ratio.
3. **Shaping that outcompetes the task.** All dense reward is potential-based
   (see ``grasprl/envs/reward.py``), and ``load_run_config`` refuses a config
   that sets ``reward.gamma``, since it must equal ``ppo.gamma`` for the shaping
   to stay policy-invariant.

Four leashes keep the policy near its imitation prior: a tight ``clip_coef``, a
KL penalty against a frozen copy, early stopping on ``approx_kl``, and a low
learning rate with few epochs and gradient accumulation.

Train::

    MUJOCO_GL=egl uv run python -m grasprl.ppo.train_ppo \
        --checkpoint checkpoints/base_smolvla --out runs/ppo_seed0 --seed 0

Writes ``runs/<run>/metrics.jsonl`` (one row per update) and
``checkpoints/<NNNNNN>/pretrained_model/``, a plain LeRobot checkpoint that
``grasprl.eval.evaluate`` and ``grasprl.real.run`` load unchanged.

Note that ``exploration is not free``: the SDE sampler scores lower than the
deterministic ODE it perturbs, so evaluation always uses the ODE.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from grasprl.envs.pickplace_env import DEFAULT_TASK, EnvConfig
from grasprl.envs.reward import RewardConfig
from grasprl.envs.vec_env import VecPickPlaceEnv
from grasprl.eval.stats import wilson
from grasprl.policy.heads import build_value_head
from grasprl.policy.loader import (
    load_smolvla,
    observations_to_batch,
    postprocess_actions,
)
from grasprl.policy.smolvla_flow_sde import (
    FlowSDEConfig,
    FlowSDESampler,
    chain_kl,
)

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class PPOConfig:
    total_env_steps: int = 300000
    rollout_steps: int = 32
    epochs: int = 2
    minibatch_size: int = 8
    grad_accum: int = 4
    lr: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.015     # per scored coordinate, see the module docstring
    kl_coef: float = 5.0         # multiplies a per-coordinate KL, hence the scale
    value_head_lr: float = 3e-4
    normalize_advantages: bool = True
    seed: int = 0
    log_every: int = 1
    save_every: int = 10


@dataclass
class RunConfig:
    sampler: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    reward: dict = field(default_factory=dict)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def load_run_config(path: str | Path | None) -> RunConfig:
    raw = yaml.safe_load(Path(path or _CONFIGS / "ppo_flow_sde.yaml").read_text())
    reward = dict(raw.get("reward", {}))
    if "gamma" in reward:
        raise ValueError(
            "reward.gamma is set from ppo.gamma; setting it separately would break "
            "the policy invariance of the potential-based shaping"
        )
    return RunConfig(sampler=raw.get("sampler", {}), env=raw.get("env", {}),
                     reward=reward, ppo=PPOConfig(**raw.get("ppo", {})))


# ---------------------------------------------------------------------------
# Rollout buffer and advantage estimation
# ---------------------------------------------------------------------------

class Buffer:
    """Flat store of one rollout, in collection order.

    Observations are kept as raw uint8 frames rather than as encoded features:
    the frozen prefix has to be recomputed under ``no_grad`` each epoch anyway,
    and the KV cache it produces is far larger than the images that generate it.
    """

    def __init__(self):
        self.front: list[np.ndarray] = []
        self.wrist: list[np.ndarray] = []
        self.state: list[np.ndarray] = []
        self.chain: list[np.ndarray] = []
        self.log_prob: list[np.ndarray] = []
        self.value: list[np.ndarray] = []
        self.reward: list[np.ndarray] = []
        self.done: list[np.ndarray] = []

    def add(self, obs, chain, log_prob, value, reward, done) -> None:
        for i, o in enumerate(obs):
            self.front.append(o["front"])
            self.wrist.append(o["wrist"])
            self.state.append(o["state"])
            self.chain.append(chain[i])
            self.log_prob.append(log_prob[i])
        self.value.append(value)
        self.reward.append(reward)
        self.done.append(done)

    def observations(self, idx) -> list[dict]:
        return [{"front": self.front[i], "wrist": self.wrist[i], "state": self.state[i]}
                for i in idx]

    def __len__(self) -> int:
        return len(self.front)


def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Generalized advantage estimation on the outer (environment) MDP.

    ``rewards``, ``values`` and ``dones`` are ``(T, n_envs)``. ``dones[t]`` marks
    that the episode ended during step ``t``, so its bootstrap is dropped. Time
    limits are handled before this is called, by folding the discounted value of
    the truncated observation into that step's reward, which is why truncation
    needs no special case here.
    """
    t_steps, n_envs = rewards.shape
    adv = np.zeros_like(rewards)
    gae = np.zeros(n_envs, dtype=np.float32)
    next_value = last_value
    for t in reversed(range(t_steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]
    return adv, adv + values


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, task: str, checkpoint: str, out: Path, run: RunConfig,
                 device: torch.device):
        self.task = task
        self.cfg = run.ppo
        self.out = out
        self.device = device

        self.policy, self.pre, self.post = load_smolvla(checkpoint, device)
        self.sampler = FlowSDESampler(
            self.policy, FlowSDEConfig(**run.sampler),
            n_exec=int(run.env.get("n_exec", 10)),
        ).to(device)
        self.sampler.freeze_backbone()
        self.value_head = build_value_head(self.sampler).to(device)

        # Frozen reference for the KL leash. Only its action expert is ever used:
        # the VLM half is identical to the trained policy's (both frozen), so the
        # reference reuses the prefix the trained sampler already computed and
        # costs one extra expert pass per minibatch.
        self.reference = copy.deepcopy(self.sampler).to(device)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()

        self.optimizer = torch.optim.AdamW([
            {"params": self.sampler.trainable_parameters(), "lr": self.cfg.lr},
            {"params": self.value_head.parameters(), "lr": self.cfg.value_head_lr},
        ], eps=1e-5)

        env_cfg = EnvConfig(
            n_exec=int(run.env.get("n_exec", 10)),
            max_ticks=int(run.env.get("max_ticks", 300)),
            domain_randomize=bool(run.env.get("domain_randomize", True)),
            reward=RewardConfig(gamma=self.cfg.gamma, **run.reward),
        )
        self.n_exec = env_cfg.n_exec
        self.n_envs = int(run.env.get("n_envs", 4))
        self.envs = VecPickPlaceEnv(self.n_envs, cfg=env_cfg, seed=self.cfg.seed,
                                    task=task)

    # -- helpers -------------------------------------------------------------

    def encode(self, observations):
        batch = observations_to_batch(observations, self.task,
                                      self.pre, self.device)
        return batch, self.sampler.encode(batch)

    @torch.no_grad()
    def values_of(self, observations) -> np.ndarray:
        batch, prefix = self.encode(observations)
        v = self.value_head(prefix.pooled, self.policy.prepare_state(batch))
        return v.cpu().numpy()

    # -- rollout -------------------------------------------------------------

    def collect(self, obs) -> tuple[Buffer, list, dict, list]:
        """Run ``rollout_steps`` decisions in every env and return the buffer."""
        buf = Buffer()
        finished: list[dict] = []
        truncations: list[tuple[int, int, dict]] = []   # (t, env index, terminal obs)
        ticks = 0

        for t in range(self.cfg.rollout_steps):
            with torch.no_grad():
                batch, prefix = self.encode(obs)
                chain_out = self.sampler.rollout(prefix)
                values = self.value_head(prefix.pooled, self.policy.prepare_state(batch))
            actions = postprocess_actions(chain_out.actions[:, :self.n_exec], self.post)

            next_obs, rewards, dones, infos = self.envs.step(actions)
            buf.add(obs, chain_out.chain.cpu().numpy(),
                    chain_out.log_prob.cpu().numpy(), values.cpu().numpy(),
                    rewards.copy(), dones.astype(np.float32))
            ticks += int(sum(i["step_ticks"] for i in infos))

            for i, done in enumerate(dones):
                if not done:
                    continue
                finished.append({"t": t, "env": i, **infos[i]["final_info"]})
                if infos[i]["truncated"]:
                    truncations.append((t, i, infos[i]["final_obs"]))
            obs = next_obs

        return buf, finished, {"ticks": ticks, "truncations": truncations}, obs

    # -- update --------------------------------------------------------------

    def update(self, buf: Buffer, advantages: np.ndarray, returns: np.ndarray) -> dict:
        cfg = self.cfg
        n = len(buf)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        # buf.log_prob (from collection) is deliberately NOT used as the PPO
        # denominator; see the note below on batch-shape sensitivity. It is kept
        # only as a diagnostic of how far the two forward paths disagree.
        collected_lp = torch.as_tensor(np.asarray(buf.log_prob), dtype=torch.float32,
                                       device=self.device)
        chains = torch.as_tensor(np.asarray(buf.chain), dtype=torch.float32)
        if cfg.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n_scored = self.sampler.n_scored
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        logs = {"pg_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                "kl_reference": 0.0, "clip_fraction": 0.0}
        # Sanity probe. On the very first minibatch of an update no optimizer
        # step has run yet, so the policy is bit-identical to the one that
        # collected the data: the importance ratio must be exactly 1 and this
        # must be ~0. If it is not, the replay is not reproducing collection
        # (a prefix, chain or indexing mismatch) and every ratio in the update
        # is meaningless -- which looks exactly like a leash that is too tight.
        first_kl = float("nan")
        n_batches = 0
        stop = False
        self.optimizer.zero_grad(set_to_none=True)

        # One fixed partition for the whole update, so the behaviour log-probs
        # below are recomputed on exactly the batches the epochs will use.
        order = np.random.permutation(n)
        parts = [order[s:s + cfg.minibatch_size] for s in range(0, n, cfg.minibatch_size)]

        # Recompute the behaviour log-probabilities under the update's OWN
        # forward conditions instead of trusting the ones stored at collection.
        #
        # Collection runs at batch = n_envs; the update runs at batch =
        # minibatch_size. Different shapes make cuBLAS pick different kernels,
        # and the resulting ~1e-3 difference in the predicted velocity is enough
        # to move the summed log-probability by about a nat, because the
        # per-step std shrinks toward the end of the chain (at t=0.1 with
        # noise_scale=0.1 it is only 0.003). Measured on this setup: approx_kl
        # 1.7e-2 on the very first minibatch, before any weight had changed,
        # against a target_kl of 0.015. The early-stop therefore fired on a
        # numerical artifact, cutting 391 of 391 updates short and applying 14%
        # of the intended gradient work.
        #
        # Recomputing here makes the ratio exactly 1 at the start of the update
        # by construction, whatever the batch shapes are. Pinned by
        # tests/test_flow_sde.py::test_replay_survives_the_conditions_training_actually_uses.
        skipped = 0
        behaviour_lp, gaps = [], []
        with torch.no_grad():
            for idx in parts:
                b = observations_to_batch(buf.observations(idx),
                                          self.task, self.pre, self.device)
                pfx = self.sampler.encode(b)
                behaviour_lp.append(
                    self.sampler.rollout(pfx, chain=chains[idx].to(self.device)).log_prob)
                gaps.append((behaviour_lp[-1] - collected_lp[idx]).abs().mean().item())

        for _ in range(cfg.epochs):
            for part_i, idx in enumerate(parts):
                batch = observations_to_batch(buf.observations(idx),
                                              self.task, self.pre, self.device)
                chain = chains[idx].to(self.device)

                with torch.no_grad():
                    prefix = self.sampler.encode(batch)
                    ref_out = self.reference.rollout(prefix, chain=chain)
                out = self.sampler.rollout(prefix, chain=chain)

                # Ratio per inner denoise step; the outer advantage is shared
                # across the K inner steps of one decision.
                mb_adv = adv_t[idx].unsqueeze(1).expand_as(out.log_prob)
                log_ratio = out.log_prob - behaviour_lp[part_i]

                # Check the trust region BEFORE spending a backward pass on this
                # minibatch, and bail without contributing anything if it is out.
                #
                # The guard used to run after backward(), which let a blown-up
                # minibatch poison the accumulated gradient before anyone looked.
                # That is not hypothetical here: one action is a 10 x 6 block
                # scored through a 10-step chain whose per-step std falls to
                # 0.003, so log-probabilities are sums over 60 sensitive
                # coordinates and the ratio is correspondingly explosive.
                # Measured on a 32-update run: approx_kl of 22.3 (log_ratio ~ 52,
                # a ratio of e^52) reaching pg_loss = 918, twice more above 100.
                # Clipping does not save it -- the surrogate takes
                # max(-adv*ratio, -adv*clip(ratio)), which selects the UNclipped
                # branch whenever the advantage is negative.
                with torch.no_grad():
                    probe_kl = float(((log_ratio.exp() - 1) - log_ratio).mean()
                                      / n_scored)
                if cfg.target_kl and probe_kl > cfg.target_kl:
                    skipped += 1
                    stop = True
                    break

                # Belt and braces: even inside the trust region, exponentiating a
                # 60-coordinate sum can overflow fp32. Bounded well above
                # anything clip_coef admits, so it never binds in normal use.
                ratio = log_ratio.clamp(-10.0, 10.0).exp()
                pg = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()

                value = self.value_head(prefix.pooled,
                                        self.policy.prepare_state(batch))
                v_loss = 0.5 * ((value - ret_t[idx]) ** 2).mean()

                # Exact KL to the imitation policy on the scored coordinates: the
                # two share the per-step std, so it reduces to a scaled squared
                # distance of the transition means.
                kl_ref = torch.stack([
                    chain_kl(self.sampler.scored_block(out.means[:, k]),
                             self.sampler.scored_block(ref_out.means[:, k]),
                             out.stds[k])
                    for k in range(out.means.shape[1])
                ], dim=1).mean() / n_scored

                entropy = out.entropy.mean() / n_scored
                loss = (pg + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                        + cfg.kl_coef * kl_ref)
                (loss / cfg.grad_accum).backward()
                n_batches += 1

                if n_batches % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    # Schulman's low-variance KL estimator, non-negative by
                    # construction, so a spike is unambiguous. Per coordinate.
                    approx_kl = (((ratio - 1) - log_ratio).mean() / n_scored).item()
                    clip_frac = ((ratio - 1).abs() > cfg.clip_coef).float().mean().item()
                if n_batches == 1:
                    first_kl = approx_kl
                logs["pg_loss"] += pg.item()
                logs["value_loss"] += v_loss.item()
                logs["entropy"] += entropy.item()
                logs["approx_kl"] += approx_kl
                logs["kl_reference"] += kl_ref.item()
                logs["clip_fraction"] += clip_frac

                if cfg.target_kl and approx_kl > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break

        # Flush a partial accumulation so the last minibatches are not discarded.
        if n_batches % cfg.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        out_logs = {k: v / max(n_batches, 1) for k, v in logs.items()}
        out_logs["minibatches"] = n_batches
        out_logs["early_stopped"] = int(stop)
        out_logs["first_minibatch_kl"] = first_kl
        # How far the collection-time forward and the update-time forward
        # disagree, in nats of summed log-probability. Not used in the loss;
        # logged so the artifact that once throttled this run stays visible.
        out_logs["replay_lp_gap"] = float(np.mean(gaps)) if gaps else float("nan")
        out_logs["skipped_minibatches"] = skipped
        out_logs["optimizer_steps"] = n_batches // cfg.grad_accum
        return out_logs

    # -- checkpointing -------------------------------------------------------

    def save(self, path: Path, keep_last: int = 3) -> None:
        """Write a checkpoint the evaluator loads like any LeRobot checkpoint."""
        model_dir = path / "pretrained_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(model_dir)
        self.pre.save_pretrained(model_dir)
        self.post.save_pretrained(model_dir)
        torch.save(self.value_head.state_dict(), path / "value_head.pt")
        # SmolVLA is about 1.8 GB on disk, so keep only the most recent few.
        numbered = sorted(p for p in path.parent.glob("[0-9]" * 6) if p.is_dir())
        for old in numbered[:-keep_last]:
            shutil.rmtree(old, ignore_errors=True)

    def close(self):
        self.envs.close()


def train(
    checkpoint: str,
    out_dir: str,
    task: str = DEFAULT_TASK,
    config_path: str | None = None,
    seed: int | None = None,
    total_env_steps: int | None = None,
) -> dict:
    run = load_run_config(config_path)
    cfg = run.ppo
    if seed is not None:
        cfg.seed = seed
    if total_env_steps is not None:
        cfg.total_env_steps = total_env_steps

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"task": task, "checkpoint": checkpoint, "sampler": run.sampler,
         "env": run.env, "reward": run.reward, "ppo": cfg.__dict__}, indent=2))
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = Trainer(task, checkpoint, out, run, device)
    n_trainable = sum(p.numel() for p in tr.sampler.trainable_parameters())
    print(f"device={device} envs={tr.n_envs} K={tr.sampler.num_steps} "
          f"n_exec={tr.n_exec} trainable={n_trainable / 1e6:.1f}M "
          f"seed={cfg.seed}", flush=True)

    obs = tr.envs.reset()
    env_steps = update = 0
    returns_hist: list[float] = []
    success_hist: list[float] = []
    # The point of this run is grasp stability, so the slip rate is logged beside
    # the success rate: a method can raise success by grabbing more often without
    # holding on any better, and that would be the wrong win.
    slip_hist: list[float] = []
    lifted_hist: list[float] = []
    running = np.zeros(tr.n_envs, dtype=np.float32)
    best_score, best_retention, best_update = -1.0, 0.0, 0
    t_start = time.time()

    while env_steps < cfg.total_env_steps:
        update += 1
        buf, finished, meta, obs = tr.collect(obs)
        env_steps += meta["ticks"]

        rewards = np.asarray(buf.reward, dtype=np.float32)
        values = np.asarray(buf.value, dtype=np.float32)
        dones = np.asarray(buf.done, dtype=np.float32)

        # Time-limit correction: a truncated episode is not really over, so fold
        # the discounted value of its final observation into that step's reward
        # before the done mask cuts the bootstrap. Without this, running out of
        # clock looks exactly like failing, and the policy learns to be timid.
        if meta["truncations"]:
            terminal_values = tr.values_of([o for _, _, o in meta["truncations"]])
            for (t, i, _), v in zip(meta["truncations"], terminal_values, strict=True):
                rewards[t, i] += cfg.gamma * float(v)

        last_value = tr.values_of(obs)
        advantages, returns = compute_gae(rewards, values, dones, last_value,
                                          cfg.gamma, cfg.gae_lambda)
        stats = tr.update(buf, advantages.reshape(-1), returns.reshape(-1))

        # Episode bookkeeping: rewards are per outer step, so accumulate them and
        # flush whenever an env finished during this rollout.
        for t in range(cfg.rollout_steps):
            running += rewards[t]
            for ep in [e for e in finished if e["t"] == t]:
                i = ep["env"]
                returns_hist.append(float(running[i]))
                success_hist.append(float(ep["success"]))
                slip_hist.append(float(ep["category"] == "grasp_slip"))
                lifted_hist.append(float(ep["ever_lifted"]))
                running[i] = 0.0

        recent = slice(-20, None)
        row = {
            "update": update,
            "env_steps": env_steps,
            "episodes": len(returns_hist),
            "success_rate": float(np.mean(success_hist[recent])) if success_hist else 0.0,
            "episode_return": float(np.mean(returns_hist[recent])) if returns_hist else 0.0,
            "slip_rate": float(np.mean(slip_hist[recent])) if slip_hist else 0.0,
            "lift_rate": float(np.mean(lifted_hist[recent])) if lifted_hist else 0.0,
            "steps_per_sec": env_steps / max(time.time() - t_start, 1e-6),
            "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 2**30
                            if torch.cuda.is_available() else 0.0),
            **stats,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if update % cfg.log_every == 0:
            print(
                f"upd {update:>4} | ticks {env_steps:>7} | eps {row['episodes']:>4} "
                f"| success {row['success_rate']:.2f} | ret {row['episode_return']:7.2f} "
                f"| slip {row['slip_rate']:.2f} | lift {row['lift_rate']:.2f} "
                f"| pg {stats['pg_loss']:+.4f} "
                f"| v {stats['value_loss']:7.3f} | kl_ref {stats['kl_reference']:.4f} "
                f"| kl {stats['approx_kl']:.4f} | kl0 {stats['first_minibatch_kl']:.5f} "
                f"| lpgap {stats['replay_lp_gap']:.2f} "
                f"| skip {stats['skipped_minibatches']} "
                f"| mb {stats['minibatches']:>2} | clip {stats['clip_fraction']:.2f} "
                f"| {row['steps_per_sec']:.1f} ticks/s",
                flush=True,
            )
        if update % cfg.save_every == 0:
            tr.save(out / "checkpoints" / f"{update:06d}")

        # Keep the best policy, not just the most recent one. This run's
        # characteristic failure is to improve fast and then run away from the
        # imitation prior and collapse: retention reached ~0.70 by update 6 and
        # was 0.00 by update 13. Without this, `last/` is the wreckage.
        #
        # Scored on retention (success among episodes that lifted a cube), which
        # is the metric the comparison reports, and only once enough episodes
        # have accumulated for the rolling window to mean anything.
        if len(returns_hist) >= 20:
            n_lift = int(sum(lifted_hist[-20:]))
            n_succ = int(sum(success_hist[-20:]))
            # Scored on the WILSON LOWER BOUND of retention, not the raw ratio.
            # The raw ratio has no sample size in it, so a window containing a
            # single lifted episode that happened to succeed scores 1.000 and
            # beats a genuinely better policy measured over seven. That is not
            # hypothetical: it happened, and 1/1 at update 32 overwrote 4/7 at
            # update 13. The lower bound ranks 4/7 (0.250) above 1/1 (0.207),
            # which is the ordering that reflects what is actually known.
            score = wilson(n_succ, n_lift).lo if n_lift else 0.0
            if score > best_score:
                best_score, best_update = score, update
                best_retention = n_succ / n_lift
                tr.save(out / "checkpoints" / "best", keep_last=10**9)
                print(f"       new best: retention {n_succ}/{n_lift} "
                      f"= {best_retention:.3f} (lower bound {score:.3f}) "
                      f"at update {update} -> checkpoints/best", flush=True)

    tr.save(out / "checkpoints" / "last")
    tr.close()
    summary = {
        "updates": update,
        "env_steps": env_steps,
        "episodes": len(returns_hist),
        "final_success_rate": float(np.mean(success_hist[-20:])) if success_hist else 0.0,
        "best_retention": best_retention,
        "best_score": best_score,
        "best_update": best_update,
        "final_slip_rate": float(np.mean(slip_hist[-20:])) if slip_hist else 0.0,
        "wall_clock_s": time.time() - t_start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done: {summary}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/base_smolvla",
                   help="frozen SmolVLA pretrained_model/ directory to start from")
    p.add_argument("--out", default=None, help="default runs/ppo_seed<seed>")
    p.add_argument("--config", default=None, help="default configs/ppo_flow_sde.yaml")
    p.add_argument("--task", default=DEFAULT_TASK, help="language instruction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-env-steps", type=int, default=None)
    args = p.parse_args(argv)

    out = args.out or f"runs/ppo_seed{args.seed}"
    train(checkpoint=args.checkpoint, out_dir=out, task=args.task,
          config_path=args.config, seed=args.seed,
          total_env_steps=args.total_env_steps)


if __name__ == "__main__":
    main()
