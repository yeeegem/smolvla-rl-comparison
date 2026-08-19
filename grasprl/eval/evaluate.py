"""Score one arm of the comparison in the calibrated contact sim.

The headline number is success rate, but on its own it cannot answer the
question this repo asks. A method could raise success by grasping more often
while holding on no better, or by holding on better while attempting fewer
grasps. So every run reports the full first-cause failure breakdown using the
same category names the real harness records, plus the two rates that separate
those cases:

* ``acquisition`` -- fraction of episodes where a cube left the table at all;
* ``retention``   -- of those, the fraction that reached the cup. **This is the
  headline**: it is the grasp-slip question, and the stage that responds to the
  physics both RL methods are trying to improve. Success rate mixes the two and
  is dominated by acquisition, which barely responds to anything.

The comparison is decided in sim, against the sim baseline. Real-arm rates are
printed for reference only.

Rates carry 95% Wilson intervals computed from pooled counts. Episodes are
i.i.d., so extra seeds are just extra episodes and are pooled rather than
averaged -- one seed of 200 episodes and two of 100 are statistically the same
thing, and a between-seed spread over two groups would be a much worse error bar
than the binomial interval. Retention's denominator is only the episodes that
lifted, so its interval is about three times wider than success rate's; that
width is the reason a head-to-head needs more episodes than it first appears.

Seeds are split into a **validation** range and a disjoint **held-out** range.
Guided Action Flow's hyperparameters are tuned on validation and reported on
held-out; re-tuning on held-out would make it a second validation set. The
paper this repo reproduces makes exactly that mistake its central warning (a
+10.0 pp validation gain that was +2.5 pp held out), so the split is enforced
here rather than left to discipline.

Usage::

    MUJOCO_GL=egl uv run python -m grasprl.eval.evaluate \\
        --method base --checkpoint checkpoints/base_smolvla \\
        --label base --episodes 200 --split heldout
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

from grasprl.envs import rules
from grasprl.envs.pickplace_env import DEFAULT_TASK, EnvConfig
from grasprl.envs.vec_env import VecPickPlaceEnv
from grasprl.eval.stats import wilson
from grasprl.policy.actor import build_actor

# Disjoint env-seed ranges. Any overlap would let a hyperparameter chosen on
# validation be reported on the very layouts it was chosen for.
SPLIT_BASE = {"validation": 100_000, "heldout": 700_000, "train": 0}


def evaluate(
    method: str = "base",
    checkpoint: str = "checkpoints/base_smolvla",
    label: str | None = None,
    episodes: int = 100,
    seeds: tuple[int, ...] = (0,),
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
    progress_every: int = 20,
    stochastic: bool = False,
    noise_scale: float = 0.2,
    num_steps: int | None = None,
) -> dict:
    if split not in SPLIT_BASE:
        raise ValueError(f"split must be one of {sorted(SPLIT_BASE)}, got {split!r}")
    label = label or method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = build_actor(method, checkpoint, device, task, n_exec=n_exec,
                        critic_dir=critic_dir, guidance=guidance, seed=seeds[0],
                        stochastic=stochastic, noise_scale=noise_scale,
                        num_steps=num_steps)
    if stochastic and not quiet:
        print(f"SDE sampler probe: noise_scale={noise_scale} "
              f"num_steps={num_steps or actor.sampler.num_steps} "
              f"(this is what PPO explores with, not what it is scored with)")

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
        t_start = time.perf_counter()
        next_report = progress_every
        while len(done_eps) < episodes:
            actions = actor.act(obs)
            obs, _r, dones, infos = envs.step(actions)
            for i, d in enumerate(dones):
                if d and len(done_eps) < episodes:
                    done_eps.append(infos[i]["final_info"])
            # A 200-episode run is ~15 minutes of silence otherwise. Reporting
            # only per seed was fine at three seeds and is useless at one, which
            # is the default now that seeds are pooled rather than averaged.
            if not quiet and progress_every and len(done_eps) >= next_report:
                next_report = len(done_eps) + progress_every
                n_done = len(done_eps)
                elapsed = time.perf_counter() - t_start
                eta = elapsed / n_done * (episodes - n_done)
                lifted = sum(e["ever_lifted"] for e in done_eps)
                won = sum(e["success"] for e in done_eps)
                ret = f"{won}/{lifted} ({won / lifted:.2f})" if lifted else "-/0"
                print(f"  [{n_done:>4}/{episodes}]  retention {ret:<14}"
                      f"acquisition {lifted / n_done:.2f}  success {won / n_done:.2f}"
                      f"  | {elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
                      flush=True)
        envs.close()
        per_seed.append(_aggregate(done_eps))
        if not quiet:
            s = per_seed[-1]
            print(f"  seed {seed}: retention {s['retention_rate']:.3f} "
                  f"({s['n_success']}/{s['n_lifted']}) "
                  f"acquisition {s['acquisition_rate']:.3f} "
                  f"| success {s['success_rate']:.3f} "
                  f"slip {s['rates']['grasp_slip']:.3f} "
                  f"nothing {s['rates']['grabbed_nothing']:.3f}", flush=True)

    out = _combine(per_seed)
    out.update({"label": label, "method": method, "checkpoint": checkpoint,
                "critic": critic_dir, "split": split, "seeds": list(seeds),
                "episodes_per_seed": episodes, "n_exec": n_exec,
                "domain_randomize": domain_randomize, "per_seed": per_seed})
    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        Path(results_dir, f"eval_{label}.json").write_text(json.dumps(out, indent=2))
    if not quiet:
        r, a, s = out["retention"], out["acquisition"], out["success"]
        print(f"\n{label}  ({out['episodes_total']} episodes, 95% Wilson intervals)")
        print(f"  RETENTION    {r['rate']:6.1%}  [{r['ci_low']:.1%}, {r['ci_high']:.1%}]"
              f"   {r['successes']}/{r['trials']} lifted episodes kept   <- headline")
        print(f"  acquisition  {a['rate']:6.1%}  [{a['ci_low']:.1%}, {a['ci_high']:.1%}]"
              f"   {a['successes']}/{a['trials']}")
        print(f"  success      {s['rate']:6.1%}  [{s['ci_low']:.1%}, {s['ci_high']:.1%}]"
              f"   {s['successes']}/{s['trials']}")
        print("  real arm     retention 64.0% (16/25), acquisition 83.3% (25/30)")
        print("  " + "  ".join(f"{c}={v:.0%}" for c, v in out["rates_mean"].items() if v))
    return out


def _aggregate(eps: list[dict]) -> dict:
    """Per-seed record. Keeps raw COUNTS, because seeds are pooled, not averaged.

    Episodes are i.i.d. draws, so the uncertainty on every rate here is binomial
    and computable from the counts. Averaging rates across a couple of seeds and
    reporting their spread would be a far noisier estimate of the same thing, and
    it cannot be pooled afterwards.
    """
    n = max(len(eps), 1)
    counts = {c: sum(e["category"] == c for e in eps) for c in rules.CATEGORIES}
    chosen = [e["cube_chosen"] for e in eps if e["cube_chosen"]]
    p_left = (sum(c == "left" for c in chosen) / len(chosen)) if chosen else 0.5
    offs = [e["grasp_offset_m"] for e in eps if np.isfinite(e["grasp_offset_m"])]
    lifted = sum(e["ever_lifted"] for e in eps)
    succeeded = sum(e["success"] for e in eps)
    return {
        "episodes": len(eps),
        # The task is two stages that fail independently, and calibration showed
        # they are calibrated to very different degrees: acquisition ~0.34
        # against the real arm's 0.83, retention ~0.37 against 0.64. So they are
        # reported separately, and RETENTION is the headline -- it is the
        # grasp-slip question, and the half the sim gets closest to right.
        "n_success": int(succeeded),
        "n_lifted": int(lifted),                 # retention's denominator
        "n_gripped": int(sum(e["ever_gripped"] for e in eps)),
        "counts": counts,
        "success_rate": succeeded / n,
        "acquisition_rate": lifted / n,
        "retention_rate": (succeeded / lifted) if lifted else float("nan"),
        "lift_rate": lifted / n,                 # alias, kept for older readers
        "grip_rate": sum(e["ever_gripped"] for e in eps) / n,
        "rates": {c: v / n for c, v in counts.items()},
        "mode_balance": abs(p_left - 0.5),
        "n_chosen_left": int(sum(c == "left" for c in chosen)),
        "n_chosen": len(chosen),
        "mean_ticks": float(np.mean([e["ticks"] for e in eps])) if eps else 0.0,
        "mean_grasp_offset_m": float(np.mean(offs)) if offs else float("nan"),
    }


def _combine(per_seed: list[dict]) -> dict:
    """Pool the seeds into one set of counts and attach Wilson intervals."""
    total = sum(s["episodes"] for s in per_seed)
    n_success = sum(s["n_success"] for s in per_seed)
    n_lifted = sum(s["n_lifted"] for s in per_seed)
    counts = {c: sum(s["counts"][c] for s in per_seed) for c in rules.CATEGORIES}

    success = wilson(n_success, total)
    acquisition = wilson(n_lifted, total)
    # Retention's denominator is the episodes that got a cube up, not all of
    # them, so its interval is roughly three times wider than success rate's at
    # the current acquisition rate. That is the honest width, and it is the
    # reason a head-to-head needs many more episodes than it looks like.
    retention = wilson(n_success, n_lifted)
    slip = wilson(counts["grasp_slip"], total)
    chosen = sum(s["n_chosen"] for s in per_seed)
    left = sum(s["n_chosen_left"] for s in per_seed)

    def g(key: str) -> np.ndarray:
        return np.array([s[key] for s in per_seed], float)

    return {
        "episodes_total": total,
        "success": success.to_dict(),
        "acquisition": acquisition.to_dict(),
        "retention": retention.to_dict(),
        "slip": slip.to_dict(),
        # Flat aliases so the report and older result files agree.
        "success_mean": success.rate, "success_std": success.half_width,
        "acquisition_mean": acquisition.rate, "acquisition_std": acquisition.half_width,
        "retention_mean": retention.rate, "retention_std": retention.half_width,
        "slip_mean": slip.rate, "slip_std": slip.half_width,
        "lift_mean": acquisition.rate,
        "grip_mean": sum(s["n_gripped"] for s in per_seed) / max(total, 1),
        "counts": counts,
        # Pooled over every scored trial, not averaged over seeds: a seed with
        # two successes would otherwise weigh as much as one with forty.
        "mode_balance_mean": abs((left / chosen if chosen else 0.5) - 0.5),
        "rates_mean": {c: v / max(total, 1) for c, v in counts.items()},
        "mean_ticks": float(np.average(g("mean_ticks"),
                                       weights=g("episodes"))) if per_seed else 0.0,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="base", choices=["base", "ppo", "gaf"])
    p.add_argument("--checkpoint", default="checkpoints/base_smolvla")
    p.add_argument("--critic", default=None, help="trained critic dir, for --method gaf")
    p.add_argument("--label", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Episodes are i.i.d. and seeds are POOLED, not averaged, so one "
                        "seed with N episodes and two with N/2 are statistically the same. "
                        "Extra seeds only vary the sampler-noise stream.")
    p.add_argument("--split", default="heldout", choices=sorted(SPLIT_BASE))
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-exec", type=int, default=10)
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--no-domain-randomize", action="store_true")
    p.add_argument("--beta", type=float, default=None, help="GAF guidance strength")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--clip-norm", type=float, default=None)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--progress-every", type=int, default=20,
                   help="print a running tally every N episodes (0 to disable)")
    p.add_argument("--stochastic", action="store_true",
                   help="score the SDE sampler PPO explores with, instead of the "
                        "deterministic ODE it is evaluated with. The gap is the "
                        "exploration tax; if it drives success to zero, PPO has no "
                        "positive reward to learn from.")
    p.add_argument("--noise-scale", type=float, default=0.2,
                   help="SDE exploration noise, with --stochastic")
    p.add_argument("--num-steps", type=int, default=None,
                   help="flow steps (default: the checkpoint's own, 10)")
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
             guidance=guidance, results_dir=a.results_dir,
             progress_every=a.progress_every, stochastic=a.stochastic,
             noise_scale=a.noise_scale, num_steps=a.num_steps)


if __name__ == "__main__":
    main()
