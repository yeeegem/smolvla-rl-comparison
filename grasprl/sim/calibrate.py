"""Pick the sim's grasp parameters, and check the sim is fit to compare methods in.

**What this is for.** The deliverable of this repo is a comparison of two RL
methods against one baseline **in simulation**. The sim does not have to
reproduce the real SO-ARM101; it has to be a *fair and discriminative testbed*.
The real-arm profile below is quoted throughout as a reference point, because it
is why grasp slip is the failure being targeted at all, but matching it is not
the acceptance criterion and a sweep that misses it is not a failure.

What the sim actually has to satisfy, and what ``sweep`` checks:

1. **Slip has to be possible.** In the source repo the cube is welded to the
   gripper, so a grasp cannot fail and there is nothing to learn. The contact
   pads fix that; ``gripper_forcerange`` is the knob that decides how often a
   held cube is dropped.
2. **Retention has to sit away from the floor and the ceiling.** It is the
   headline metric, and a baseline pinned at 0.05 or 0.95 leaves no room for a
   method to show an effect, and measures a proportion at its least sensitive.
   Mid-range is what discriminates.
3. **Enough episodes have to reach the retention stage.** Retention is measured
   only over episodes that got a cube off the table, so the acquisition rate is
   the sample-efficiency of the whole experiment. At acquisition 0.34 a
   200-episode run yields only ~68 retention samples.

Two knobs, close to independent:

* ``capture_radius`` -- acquisition tolerance (see ``Scene.capture``). Measured
  over 20 to 45 mm it is close to inert, because the limit is the policy failing
  to close the gripper near a cube at all, not near-misses.
* ``gripper_forcerange`` -- peak servo torque, hence the pinch force holding a
  0.4 mm interference. This is the real knob: retention rises monotonically with
  it across independent rows of the grid.

``pad_friction`` is not swept by default: over mu in [0.2, 3.0] it barely moves
anything, because the pinch is torque-limited and mu*N already exceeds the 0.2 N
cube weight by two orders of magnitude. Pass ``--friction`` to sweep it anyway.

Whatever is chosen is written into ``configs/scene.yaml`` and then left alone.
Both RL methods must train and be scored on identical physics, so re-calibrating
after seeing a result would invalidate the comparison.

Usage::

    MUJOCO_GL=egl uv run python -m grasprl.sim.calibrate \\
        --checkpoint checkpoints/base_smolvla --episodes 40 --write
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import yaml

from grasprl.eval.stats import required_trials

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# Measured on the real arm. REFERENCE ONLY: this is context for why grasp slip is
# the failure being targeted, not a target the sim is required to hit. The
# comparison is decided in sim, against the sim baseline.
REAL_PROFILE = {"success": 16 / 30, "grasp_slip": 9 / 30, "grabbed_nothing": 4 / 30}

# The same 30 trials split into the two stages. 25 of 30 got a cube off the table
# (16 successes + 9 slips), and 16 of those 25 survived the carry.
REAL_ACQUISITION = 25 / 30      # got a cube airborne at all
REAL_RETENTION = 16 / 25        # of those, kept hold of it to the cup

# The sim ceiling with a weld grasp, i.e. how well the policy would do if a grasp
# never failed. Reported alongside every sweep so a cell that scores well for the
# wrong reason (e.g. by never grasping at all) is visible.
WELD_SIM_CEILING = 13 / 16


def profile_distance(result: dict) -> float:
    """L1 distance from a sim result to the real failure profile."""
    r = result["rates_mean"]
    return (abs(result["success_mean"] - REAL_PROFILE["success"])
            + abs(r["grasp_slip"] - REAL_PROFILE["grasp_slip"])
            + abs(r["grabbed_nothing"] - REAL_PROFILE["grabbed_nothing"]))


def sweep(checkpoint: str, episodes: int = 40, seeds=(0,),
          forcerange=(0.7, 0.9, 1.1, 1.4), capture=(0.020, 0.030, 0.045),
          friction=(0.6,), n_exec: int = 10, n_envs: int = 4,
          out_dir: str = "results", write: bool = False, accept_best: bool = False,
          config_path: str | None = None) -> dict:
    from grasprl.eval.evaluate import evaluate

    path = Path(config_path or _CONFIGS / "scene.yaml")
    base_cfg = yaml.safe_load(path.read_text())

    rows = []
    for cap in capture:
        for fr in forcerange:
            for mu in friction:
                cfg = copy.deepcopy(base_cfg)
                cfg["grasp"]["capture_radius"] = float(cap)
                cfg["grasp"]["gripper_forcerange"] = float(fr)
                cfg["grasp"]["pad_friction"] = [float(mu), 0.05, 0.002]
                res = evaluate(
                    method="base", checkpoint=checkpoint,
                    label=f"cal_r{cap}_F{fr}_mu{mu}",
                    episodes=episodes, seeds=tuple(seeds), split="train",
                    n_envs=n_envs, n_exec=n_exec, scene_cfg=cfg,
                    results_dir="", quiet=True,
                )
                row = {
                    "capture_radius": cap, "gripper_forcerange": fr, "pad_friction": mu,
                    "success": res["success_mean"], "slip": res["slip_mean"],
                    "grabbed_nothing": res["rates_mean"]["grabbed_nothing"],
                    "lift_rate": res["lift_mean"], "grip_rate": res["grip_mean"],
                    "distance": profile_distance(res),
                }
                rows.append(row)
                print(f"r={cap:<6} F={fr:<5} mu={mu:<4} success {row['success']:.2f} "
                      f"slip {row['slip']:.2f} nothing {row['grabbed_nothing']:.2f} "
                      f"lift {row['lift_rate']:.2f} | L1 {row['distance']:.3f}", flush=True)

    for row in rows:
        # The two stages fail independently and are what the comparison is
        # actually made of, so they are scored separately.
        row["acquisition"] = row["lift_rate"]
        row["retention"] = (row["success"] / row["lift_rate"]) if row["lift_rate"] else 0.0
        row["retention_error"] = abs(row["retention"] - REAL_RETENTION)   # reference only
        row["fitness"] = comparison_fitness(row, episodes * len(seeds))

    best = max(rows, key=lambda r: r["fitness"])
    closest_to_real = min(rows, key=lambda r: r["distance"])
    report = {
        "checkpoint": checkpoint, "episodes_per_cell": episodes, "seeds": list(seeds),
        "grid": rows, "best": best,
        "criterion": "fitness for comparing methods in sim (headroom x sample yield)",
        # Everything below is context, not an acceptance criterion.
        "reference_real_profile": REAL_PROFILE,
        "reference_real_acquisition": REAL_ACQUISITION,
        "reference_real_retention": REAL_RETENTION,
        "reference_weld_sim_ceiling": WELD_SIM_CEILING,
        "reference_closest_to_real": closest_to_real,
        "fit_for_comparison": bool(best["fitness"] > 0),
        "retention_samples_per_200_episodes": round(best["acquisition"] * 200),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "calibration.json").write_text(json.dumps(report, indent=2))

    print("\n=== calibration ===")
    print("criterion: fitness as a TESTBED for comparing the two RL methods.")
    print("The sim does not have to match the real arm; it has to leave room for a")
    print("method to show an effect, and yield enough retention samples to see it.\n")
    print(f"{'':>20}{'acquisition':>13}{'retention':>11}{'fitness':>10}")
    print(f"{'chosen cell':>20}{best['acquisition']:>13.2f}{best['retention']:>11.2f}"
          f"{best['fitness']:>10.3f}   "
          f"(r={best['capture_radius']}, F={best['gripper_forcerange']}, "
          f"mu={best['pad_friction']})")
    print(f"{'real arm (reference)':>20}{REAL_ACQUISITION:>13.2f}{REAL_RETENTION:>11.2f}")
    print(f"{'weld ceiling (ref)':>20}{'-':>13}{WELD_SIM_CEILING:>11.2f}"
          "   (grasp cannot fail)")

    n_ret = report["retention_samples_per_200_episodes"]
    print(f"\nretention samples: ~{n_ret} per 200 episodes.")
    print(f"  resolving a +15 pp retention gain needs ~{required_trials(best['retention'], 0.15)}"
          f" retention samples per arm")
    print(f"  = ~{round(required_trials(best['retention'], 0.15) / max(best['acquisition'], 1e-6))}"
          " episodes per arm")

    if not report["fit_for_comparison"]:
        print("\nNOT FIT FOR COMPARISON: retention is pinned at the floor or ceiling, or")
        print("almost nothing reaches the retention stage. Neither method could show an")
        print("effect here. Widen the grid before running them.")

    if write and (report["fit_for_comparison"] or accept_best):
        _write_grasp_values(path, best)
        why = "fit for comparison" if report["fit_for_comparison"] else "forced by --accept-best"
        print(f"\nwrote chosen values into {path} ({why})")
    elif write:
        print("\nnot fit for comparison; configs/scene.yaml left unchanged. "
              "Re-run with --accept-best to take the best cell anyway.")
    return report


def comparison_fitness(row: dict, episodes: int) -> float:
    """How useful a cell is for telling two RL methods apart. Higher is better.

    Two factors, multiplied because both are necessary:

    * **headroom** -- a baseline retention pinned near 0 or 1 leaves a method
      nowhere to move and measures a proportion where it is least sensitive.
      Peaks at 0.5 and falls off toward either end.
    * **sample yield** -- retention is only observed on episodes that lift a
      cube, so acquisition sets how many samples an episode budget actually
      buys, and therefore the power of the whole experiment.

    Deliberately says nothing about the real arm. A cell that matches the real
    failure profile but leaves no headroom is worse for this project than one
    that misses it and discriminates.
    """
    ret, acq = row["retention"], row["acquisition"]
    if acq <= 0.05 or not 0.05 < ret < 0.95:
        return 0.0
    headroom = 4 * ret * (1 - ret)     # 1.0 at ret=0.5, 0 at the extremes
    return headroom * acq


def _write_grasp_values(path: Path, best: dict) -> None:
    """Patch only the two calibrated scalars, keeping every comment in place.

    ``yaml.safe_dump`` would round-trip the file and throw away the measurement
    notes that explain why each value is what it is, which are the most useful
    thing in it.
    """
    lines = path.read_text().splitlines(keepends=True)
    out = []
    for line in lines:
        if line.startswith("  gripper_forcerange:"):
            out.append(f"  gripper_forcerange: {best['gripper_forcerange']}\n")
        elif line.startswith("  capture_radius:"):
            out.append(f"  capture_radius: {best['capture_radius']}\n")
        elif line.startswith("  pad_friction:"):
            out.append(f"  pad_friction: [{best['pad_friction']}, 0.05, 0.002]\n")
        else:
            out.append(line)
    path.write_text("".join(out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/base_smolvla")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--forcerange", type=float, nargs="+", default=[0.7, 0.9, 1.1, 1.4],
                   help="retention knob: peak gripper servo torque (Nm)")
    p.add_argument("--capture", type=float, nargs="+", default=[0.020, 0.030, 0.045],
                   help="acquisition knob: grasp tolerance radius (m); see Scene.capture")
    p.add_argument("--friction", type=float, nargs="+", default=[0.6],
                   help="pad sliding friction; measured to be a weak knob above ~0.2")
    p.add_argument("--n-exec", type=int, default=10)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--write", action="store_true",
                   help="patch the winning values into configs/scene.yaml")
    p.add_argument("--accept-best", action="store_true",
                   help="write the best cell even if the joint gate fails. Use when "
                        "retention is calibrated but acquisition is not, and record "
                        "that the sim is a relative comparison, not a real-rate predictor.")
    a = p.parse_args(argv)
    sweep(checkpoint=a.checkpoint, episodes=a.episodes, seeds=tuple(a.seeds),
          forcerange=tuple(a.forcerange), capture=tuple(a.capture),
          friction=tuple(a.friction),
          n_exec=a.n_exec, n_envs=a.n_envs, out_dir=a.out_dir, write=a.write,
          accept_best=a.accept_best)


if __name__ == "__main__":
    main()
