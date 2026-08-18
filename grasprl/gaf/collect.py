"""Collect the rollout dataset the Guided Action Flow critic is trained on.

Runs the **frozen** base policy in the calibrated contact sim and records, once
per decision, the four things Eq. (1) needs -- the policy-side state, the pooled
VLM feature, the action chunk that was actually executed, and whether the episode
had succeeded yet.

Two details are load-bearing:

* **Actions are stored in the model's own (normalised, padded) space**, taken
  straight off the sampler before the postprocessor. Guidance differentiates the
  critic with respect to ``a_hat``, which lives in exactly that space; a critic
  trained on de-normalised robot units would supply gradients in the wrong units
  and quietly scale guidance by whatever the normalisation happens to be.
* **Episodes are kept whole.** The train/validation split downstream is by
  episode, never by chunk, so the episode id travels with every sample.

Usage::

    MUJOCO_GL=egl uv run python -m grasprl.gaf.collect \\
        --checkpoint checkpoints/base_smolvla --episodes 600 \\
        --out recordings/gaf_rollouts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from grasprl.envs.pickplace_env import DEFAULT_TASK, EnvConfig
from grasprl.envs.vec_env import VecPickPlaceEnv
from grasprl.policy.loader import load_smolvla, observations_to_batch, postprocess_actions
from grasprl.policy.smolvla_flow_sde import FlowSDEConfig, FlowSDESampler


def collect(
    checkpoint: str,
    out_dir: str,
    episodes: int = 600,
    n_envs: int = 4,
    n_exec: int = 10,
    horizon: int = 10,
    max_ticks: int = 300,
    domain_randomize: bool = True,
    seed: int = 0,
    task: str = DEFAULT_TASK,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, post = load_smolvla(checkpoint, device)
    sampler = FlowSDESampler(policy, FlowSDEConfig(num_steps=None), n_exec=n_exec).to(device)

    envs = VecPickPlaceEnv(
        n_envs,
        cfg=EnvConfig(n_exec=n_exec, max_ticks=max_ticks, domain_randomize=domain_randomize),
        seed=seed, task=task,
    )
    gen = torch.Generator(device=device)
    gen.manual_seed(900_000 + 7919 * seed)

    # Per-env open episode buffers, flushed into the dataset on episode end.
    open_eps: list[dict] = [{"state": [], "pooled": [], "action": [], "success": []}
                            for _ in range(n_envs)]
    ds = {"state": [], "pooled": [], "action": [], "success": [], "episode": []}
    summaries: list[dict] = []
    ep_id = 0
    t0 = time.time()

    obs = envs.reset()
    while ep_id < episodes:
        with torch.no_grad():
            batch = observations_to_batch(obs, task, pre, device)
            prefix = sampler.encode(batch)
            state = policy.prepare_state(batch)
            latent = sampler.sample_ode(prefix, generator=gen)     # (B, chunk, 6)
        chunk = latent[:, :horizon, :].float().cpu().numpy()
        actions = postprocess_actions(latent[:, :n_exec], post)

        next_obs, _rewards, dones, infos = envs.step(actions)
        st = state.float().cpu().numpy()
        pl = prefix.pooled.float().cpu().numpy()

        for i in range(n_envs):
            buf = open_eps[i]
            buf["state"].append(st[i])
            buf["pooled"].append(pl[i])
            buf["action"].append(chunk[i])
            # Success is read from the env's own tracker so it means exactly what
            # it means in the results table.
            buf["success"].append(0)

            if not dones[i]:
                continue
            ep = infos[i]["final_info"]
            if ep["success"]:
                buf["success"][-1] = 1
            if ep_id < episodes:
                n = len(buf["state"])
                ds["state"].append(np.asarray(buf["state"], np.float32))
                ds["pooled"].append(np.asarray(buf["pooled"], np.float32))
                ds["action"].append(np.asarray(buf["action"], np.float32))
                ds["success"].append(np.asarray(buf["success"], np.int8))
                ds["episode"].append(np.full(n, ep_id, np.int32))
                summaries.append({"episode": ep_id, "decisions": n, **ep})
                ep_id += 1
                if ep_id % 25 == 0:
                    sr = np.mean([s["success"] for s in summaries])
                    print(f"  {ep_id}/{episodes} episodes | success so far {sr:.2f} "
                          f"| {time.time() - t0:.0f}s", flush=True)
            for k in buf:
                buf[k].clear()
        obs = next_obs

    envs.close()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "rollouts.npz",
        state=np.concatenate(ds["state"]),
        pooled=np.concatenate(ds["pooled"]),
        action=np.concatenate(ds["action"]),
        success=np.concatenate(ds["success"]),
        episode=np.concatenate(ds["episode"]),
    )
    cats: dict[str, int] = {}
    for s in summaries:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    meta = {
        "checkpoint": checkpoint, "episodes": ep_id,
        "samples": int(sum(len(a) for a in ds["state"])),
        "n_exec": n_exec, "horizon": horizon, "seed": seed,
        "domain_randomize": domain_randomize,
        "success_rate": float(np.mean([s["success"] for s in summaries])),
        "categories": cats,
        "wall_clock_s": time.time() - t0,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "episodes.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/base_smolvla")
    p.add_argument("--out", default="recordings/gaf_rollouts")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-exec", type=int, default=10)
    p.add_argument("--horizon", type=int, default=10,
                   help="H, the chunk length the critic scores; defaults to n_exec")
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--no-domain-randomize", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default=DEFAULT_TASK)
    a = p.parse_args(argv)
    collect(checkpoint=a.checkpoint, out_dir=a.out, episodes=a.episodes, n_envs=a.n_envs,
            n_exec=a.n_exec, horizon=a.horizon, max_ticks=a.max_ticks,
            domain_randomize=not a.no_domain_randomize, seed=a.seed, task=a.task)


if __name__ == "__main__":
    main()
