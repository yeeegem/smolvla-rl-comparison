"""Score one arm of the comparison in the calibrated contact sim.

The headline number is success rate, but on its own it cannot answer the
question this repo asks. A method could raise success by grasping more often
while holding on no better, or by holding on better while attempting fewer
grasps. So every run reports the full first-cause failure breakdown using the
same category names the real harness records, plus the two rates that separate
those cases:

* ``lift_rate``  -- fraction of episodes where a cube left the table at all;
* ``slip_rate``  -- fraction lost to ``grasp_slip`` specifically.

Seeds are split into a **validation** range and a disjoint **held-out** range.
Guided Action Flow's hyperparameters are tuned on validation and reported on
held-out; re-tuning on held-out would make it a second validation set. The
paper this repo reproduces makes exactly that mistake its central warning (a
+10.0 pp validation gain that was +2.5 pp held out), so the split is enforced
here rather than left to discipline.

Usage::

    MUJOCO_GL=egl uv run python -m grasprl.eval.evaluate \\
        --method base --checkpoint checkpoints/base_smolvla \\
        --label base --episodes 100 --seeds 0 1 2 --split heldout
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from grasprl.envs import rules
from grasprl.envs.pickplace_env import DEFAULT_TASK, EnvConfig
from grasprl.envs.vec_env import VecPickPlaceEnv
from grasprl.policy.actor import build_actor

# Disjoint env-seed ranges. Any overlap would let a hyperparameter chosen on
# validation be reported on the very layouts it was chosen for.
SPLIT_BASE = {"validation": 100_000, "heldout": 700_000, "train": 0}


def evaluate(
    method: str = "base",
    checkpoint: str = "checkpoints/base_smolvla",
    label: str | None = None,
    episodes: int = 100,
    seeds: tuple[int, ...] = (0, 1, 2),
    split: str = "heldout",
    n_envs: int = 4,
    n_exec: int = 10,
    max_ticks: int = 300,
    domain_randomize: bool = True,
    critic_dir: str | None = None,
    guidance=None,
    scene_cfg: dict | None = None,
    task: str = DEFAULT_TASK,
    results_dir: str = "results",
    quiet: bool = False,
) -> dict:
    if split not in SPLIT_BASE:
        raise ValueError(f"split must be one of {sorted(SPLIT_BASE)}, got {split!r}")
    label = label or method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = build_actor(method, checkpoint, device, task, n_exec=n_exec,
                        critic_dir=critic_dir, guidance=guidance, seed=seeds[0])

    per_seed = []
    for seed in seeds:
        actor.reset(seed)
        envs = VecPickPlaceEnv(
            n_envs,
            cfg=EnvConfig(n_exec=n_exec, max_ticks=max_ticks,
                          domain_randomize=domain_randomize),
            scene_cfg=scene_cfg,
            seed=SPLIT_BASE[split] + 977 * seed,
            task=task,
        )
        obs = envs.reset(seed=SPLIT_BASE[split] + 977 * seed)
        done_eps: list[dict] = []
        while len(done_eps) < episodes:
            actions = actor.act(obs)
            obs, _r, dones, infos = envs.step(actions)
            for i, d in enumerate(dones):
                if d and len(done_eps) < episodes:
                    done_eps.append(infos[i]["final_info"])
        envs.close()
        per_seed.append(_aggregate(done_eps))
        if not quiet:
            s = per_seed[-1]
            print(f"  seed {seed}: success {s['success_rate']:.3f} "
                  f"slip {s['rates']['grasp_slip']:.3f} "
                  f"nothing {s['rates']['grabbed_nothing']:.3f} "
                  f"lift {s['lift_rate']:.3f} "
                  f"|P(left)-0.5| {s['mode_balance']:.2f}", flush=True)

    out = _combine(per_seed)
    out.update({"label": label, "method": method, "checkpoint": checkpoint,
                "critic": critic_dir, "split": split, "seeds": list(seeds),
                "episodes_per_seed": episodes, "n_exec": n_exec,
                "domain_randomize": domain_randomize, "per_seed": per_seed})
    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        Path(results_dir, f"eval_{label}.json").write_text(json.dumps(out, indent=2))
    if not quiet:
        print(f"{label}: success {out['success_mean']:.3f} +/- {out['success_std']:.3f} "
              f"| slip {out['slip_mean']:.3f} | lift {out['lift_mean']:.3f}", flush=True)
    return out


def _aggregate(eps: list[dict]) -> dict:
    n = max(len(eps), 1)
    cats = {c: sum(e["category"] == c for e in eps) / n for c in rules.CATEGORIES}
    chosen = [e["cube_chosen"] for e in eps if e["cube_chosen"]]
    p_left = (sum(c == "left" for c in chosen) / len(chosen)) if chosen else 0.5
    offs = [e["grasp_offset_m"] for e in eps if np.isfinite(e["grasp_offset_m"])]
    return {
        "episodes": len(eps),
        "success_rate": sum(e["success"] for e in eps) / n,
        "lift_rate": sum(e["ever_lifted"] for e in eps) / n,
        "grip_rate": sum(e["ever_gripped"] for e in eps) / n,
        "rates": cats,
        "mode_balance": abs(p_left - 0.5),
        "mean_ticks": float(np.mean([e["ticks"] for e in eps])) if eps else 0.0,
        "mean_grasp_offset_m": float(np.mean(offs)) if offs else float("nan"),
    }


def _combine(per_seed: list[dict]) -> dict:
    g = lambda k: np.array([s[k] for s in per_seed], float)
    return {
        "success_mean": float(g("success_rate").mean()),
        "success_std": float(g("success_rate").std()),
        "lift_mean": float(g("lift_rate").mean()),
        "grip_mean": float(g("grip_rate").mean()),
        "slip_mean": float(np.mean([s["rates"]["grasp_slip"] for s in per_seed])),
        "slip_std": float(np.std([s["rates"]["grasp_slip"] for s in per_seed])),
        "rates_mean": {c: float(np.mean([s["rates"][c] for s in per_seed]))
                       for c in rules.CATEGORIES},
        "mode_balance_mean": float(g("mode_balance").mean()),
        "mean_ticks": float(g("mean_ticks").mean()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="base", choices=["base", "ppo", "gaf"])
    p.add_argument("--checkpoint", default="checkpoints/base_smolvla")
    p.add_argument("--critic", default=None, help="trained critic dir, for --method gaf")
    p.add_argument("--label", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--split", default="heldout", choices=sorted(SPLIT_BASE))
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-exec", type=int, default=10)
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--no-domain-randomize", action="store_true")
    p.add_argument("--beta", type=float, default=None, help="GAF guidance strength")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--clip-norm", type=float, default=None)
    p.add_argument("--results-dir", default="results")
    a = p.parse_args(argv)

    guidance = None
    if a.method == "gaf":
        import yaml

        from grasprl.gaf.guided_sampler import GuidanceConfig
        raw = yaml.safe_load(Path("configs/gaf.yaml").read_text()).get("guidance", {})
        for k, v in (("beta", a.beta), ("alpha", a.alpha), ("clip_norm", a.clip_norm)):
            if v is not None:
                raw[k] = v
        guidance = GuidanceConfig(**raw)

    evaluate(method=a.method, checkpoint=a.checkpoint, label=a.label,
             episodes=a.episodes, seeds=tuple(a.seeds), split=a.split,
             n_envs=a.n_envs, n_exec=a.n_exec, max_ticks=a.max_ticks,
             domain_randomize=not a.no_domain_randomize, critic_dir=a.critic,
             guidance=guidance, results_dir=a.results_dir)


if __name__ == "__main__":
    main()
