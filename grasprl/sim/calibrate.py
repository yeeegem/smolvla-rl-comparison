"""Tune the sim's grasp physics until the frozen policy fails the way it really does.

Everything in this repo is measured in the contact sim, so the contact sim has to
be worth measuring in. The target is the frozen base policy's *measured* profile
on the real SO-ARM101 (30 tier-A trials,
``sim2real-soarm-benchmark/runs/smolvla_cotrain/eval/results.csv``):

    success 53%   grasp_slip 30%   grabbed_nothing 13%

Two anchors bracket what a correct calibration must reproduce:

* With a **weld** grasp (the abstraction the demonstrations were recorded with,
  where slipping is impossible) the same policy scores **81% in sim** with a
  median reach error of 1.2 cm. So its perception and positioning transfer
  cleanly -- the sim-to-real gap that remains is almost entirely grip.
* On the **real arm** it scores 53%. The 28-point difference is the slip budget,
  and a calibrated contact sim should spend about that much.

Two knobs, one per failure mode, because they are close to independent:

* ``capture_radius`` sets **acquisition** -- how much across-the-jaws placement
  error the grasp forgives (see ``Scene.capture``). It moves ``grabbed_nothing``.
* ``gripper_forcerange`` sets **retention** -- peak servo torque, hence the
  normal force holding a 0.4 mm interference pinch. It moves ``grasp_slip``.

``pad_friction`` is deliberately not swept by default: measured over mu in
[0.2, 3.0] it barely moves anything, because the pinch is torque-limited and
mu*N already exceeds the 0.2 N cube weight by two orders of magnitude. Pass
``--friction`` to sweep it anyway.

Each cell is scored by L1 distance to the real profile over the three rates that
matter, and the winner is written back into ``configs/scene.yaml`` and then left
alone -- both RL methods must train and be scored on identical physics, so
re-calibrating after seeing a result would invalidate the comparison.

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

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# Measured on the real arm; see checkpoints/base_smolvla/PROVENANCE.md.
REAL_PROFILE = {"success": 16 / 30, "grasp_slip": 9 / 30, "grabbed_nothing": 4 / 30}

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
          out_dir: str = "results", write: bool = False,
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

    best = min(rows, key=lambda r: r["distance"])
    report = {
        "checkpoint": checkpoint, "episodes_per_cell": episodes, "seeds": list(seeds),
        "real_profile": REAL_PROFILE, "weld_sim_ceiling": WELD_SIM_CEILING,
        "grid": rows, "best": best,
        # A cell can only be trusted if the policy is actually attempting grasps
        # in it; a sim where nothing is ever lifted matches nothing meaningful.
        "gate_passed": bool(best["distance"] <= 0.24 and best["lift_rate"] >= 0.4),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "calibration.json").write_text(json.dumps(report, indent=2))

    print("\n=== calibration ===")
    print(f"real profile      : success {REAL_PROFILE['success']:.2f} "
          f"slip {REAL_PROFILE['grasp_slip']:.2f} "
          f"nothing {REAL_PROFILE['grabbed_nothing']:.2f}")
    print(f"weld sim ceiling  : success {WELD_SIM_CEILING:.2f} (grasp cannot fail)")
    print(f"best cell         : capture_radius {best['capture_radius']} "
          f"forcerange {best['gripper_forcerange']} "
          f"friction {best['pad_friction']} -> success {best['success']:.2f} "
          f"slip {best['slip']:.2f} nothing {best['grabbed_nothing']:.2f}")
    print(f"gate              : {'PASSED' if report['gate_passed'] else 'FAILED'} "
          f"(L1 {best['distance']:.3f}, lift {best['lift_rate']:.2f})")
    if not report["gate_passed"]:
        print("  The contact model does not reproduce the real failure profile.")
        print("  Widen the grid or revisit the pad geometry BEFORE running either")
        print("  RL method -- both of them are scored in this sim.")

    if write and report["gate_passed"]:
        _write_grasp_values(path, best)
        print(f"wrote calibrated values into {path}")
    elif write:
        print("gate failed; configs/scene.yaml left unchanged")
    return report


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
    a = p.parse_args(argv)
    sweep(checkpoint=a.checkpoint, episodes=a.episodes, seeds=tuple(a.seeds),
          forcerange=tuple(a.forcerange), capture=tuple(a.capture),
          friction=tuple(a.friction),
          n_exec=a.n_exec, n_envs=a.n_envs, out_dir=a.out_dir, write=a.write)


if __name__ == "__main__":
    main()
