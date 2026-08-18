"""Choose the guidance hyperparameters -- on validation seeds, once.

Guided Action Flow has four knobs (``beta``, ``clip_norm``, ``alpha``,
``min_gate``) and the paper reports it as genuinely sensitive to them: guidance
that is too strong overrules a base policy that was already competent and turns
successes into regressions. So they have to be chosen, not assumed.

They also have to be chosen *somewhere the result is not reported*. The paper's
headline caution is that its multi-family critic gained +10.0 pp on the split it
was tuned on and +2.5 pp on a locked held-out split. This module only ever
touches ``--split validation``; ``scripts/eval_all.sh`` reports the winner once
on the disjoint held-out range. The winning values are written back into
``configs/gaf.yaml`` so the reported run cannot silently use different ones.

Usage::

    MUJOCO_GL=egl uv run python -m grasprl.gaf.sweep --critic runs/gaf_critic
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import yaml

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def sweep(critic: str, checkpoint: str = "checkpoints/base_smolvla",
          episodes: int = 50, seeds=(0, 1), config_path: str | None = None,
          out_dir: str = "results", write: bool = True) -> dict:
    from grasprl.eval.evaluate import evaluate
    from grasprl.gaf.guided_sampler import GuidanceConfig

    path = Path(config_path or _CONFIGS / "gaf.yaml")
    raw = yaml.safe_load(path.read_text())
    grid = raw.get("sweep", {})
    defaults = raw.get("guidance", {})

    # The unguided baseline on the SAME validation seeds. Without it a "win" is
    # unreadable: guidance could be neutral while the seeds happen to be easy.
    base = evaluate(method="base", checkpoint=checkpoint, label="sweep_base",
                    episodes=episodes, seeds=tuple(seeds), split="validation",
                    results_dir="", quiet=True)
    print(f"unguided baseline on validation: success {base['success_mean']:.3f} "
          f"slip {base['slip_mean']:.3f}", flush=True)

    rows = []
    combos = list(itertools.product(grid.get("beta", [defaults.get("beta", 2.0)]),
                                    grid.get("alpha", [defaults.get("alpha", 10.0)]),
                                    grid.get("clip_norm", [defaults.get("clip_norm", 1.0)])))
    for beta, alpha, clip in combos:
        g = GuidanceConfig(**{**defaults, "beta": beta, "alpha": alpha, "clip_norm": clip})
        res = evaluate(method="gaf", checkpoint=checkpoint, critic_dir=critic,
                       label=f"sweep_b{beta}_a{alpha}_c{clip}", episodes=episodes,
                       seeds=tuple(seeds), split="validation", guidance=g,
                       results_dir="", quiet=True)
        row = {"beta": beta, "alpha": alpha, "clip_norm": clip,
               "success": res["success_mean"], "slip": res["slip_mean"],
               "lift": res["lift_mean"],
               "success_gain": res["success_mean"] - base["success_mean"],
               "slip_gain": base["slip_mean"] - res["slip_mean"]}
        rows.append(row)
        print(f"beta={beta:<5} alpha={alpha:<5} c={clip:<4} | success {row['success']:.3f} "
              f"({row['success_gain']:+.3f}) | slip {row['slip']:.3f} "
              f"({row['slip_gain']:+.3f})", flush=True)

    best = max(rows, key=lambda r: r["success"])
    report = {"critic": critic, "checkpoint": checkpoint, "split": "validation",
              "episodes_per_seed": episodes, "seeds": list(seeds),
              "baseline": {"success": base["success_mean"], "slip": base["slip_mean"]},
              "grid": rows, "best": best}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "gaf_sweep.json").write_text(json.dumps(report, indent=2))

    print(f"\nbest on validation: beta={best['beta']} alpha={best['alpha']} "
          f"clip={best['clip_norm']} -> success {best['success']:.3f} "
          f"({best['success_gain']:+.3f} vs unguided)")
    if best["success_gain"] <= 0:
        print("  No setting beats the unguided policy on validation. Report that as "
              "the result rather than picking the least-bad cell -- a critic that "
              "cannot help here will not help on held-out seeds either.")
    if write:
        raw["guidance"].update(beta=best["beta"], alpha=best["alpha"],
                               clip_norm=best["clip_norm"])
        _patch(path, raw["guidance"])
        print(f"locked into {path}: {raw['guidance']}")
    return report


def _patch(path: Path, guidance: dict) -> None:
    """Rewrite only the guidance scalars, preserving the file's comments."""
    out = []
    in_guidance = False
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("guidance:"):
            in_guidance = True
        elif line and not line[0].isspace() and not line.startswith("#"):
            in_guidance = False
        if in_guidance:
            for k in ("beta", "clip_norm", "alpha", "min_gate"):
                if line.startswith(f"  {k}:"):
                    tail = line.split("#", 1)
                    comment = f"    #{tail[1]}" if len(tail) > 1 else "\n"
                    line = f"  {k}: {guidance[k]}{comment}"
                    break
        out.append(line)
    path.write_text("".join(out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--critic", default="runs/gaf_critic")
    p.add_argument("--checkpoint", default="checkpoints/base_smolvla")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--no-write", action="store_true",
                   help="do not lock the winner into configs/gaf.yaml")
    a = p.parse_args(argv)
    sweep(critic=a.critic, checkpoint=a.checkpoint, episodes=a.episodes,
          seeds=tuple(a.seeds), config_path=a.config, out_dir=a.out_dir,
          write=not a.no_write)


if __name__ == "__main__":
    main()
