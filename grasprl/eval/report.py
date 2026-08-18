"""Assemble the head-to-head table this repo exists to produce.

Reads the per-arm ``results/eval_<label>.json`` written by
:mod:`grasprl.eval.evaluate` and, when they exist, the operator-scored real-arm
CSVs, and renders ``results/comparison.md``.

The table deliberately puts ``lift`` and ``grasp_slip`` next to ``success``.
"Which method improved the grasp?" is not answerable from success rate alone: a
method that grasps more often but holds no better, and a method that holds
better but attempts fewer grasps, can post the same success number and mean
opposite things.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = [("base", "frozen SmolVLA (baseline)"),
        ("ppo", "+ Flow-SDE PPO (weights updated)"),
        ("gaf", "+ Guided Action Flow (weights frozen)")]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sim_table(results_dir: Path) -> str:
    header = ("| arm | success | grasp_slip | grabbed_nothing | missed_cup | lift rate | "
              "\\|P(left)-0.5\\| |")
    rows = [header, "|---|---|---|---|---|---|---|"]
    any_row = False
    for label, name in ARMS:
        d = _load(results_dir / f"eval_{label}.json")
        if d is None:
            rows.append(f"| {name} | _not run_ | | | | | |")
            continue
        any_row = True
        r = d["rates_mean"]
        rows.append(
            f"| {name} | **{d['success_mean']:.1%}** +/- {d['success_std']:.1%} "
            f"| {r['grasp_slip']:.1%} | {r['grabbed_nothing']:.1%} "
            f"| {r['missed_cup']:.1%} | {d['lift_mean']:.1%} "
            f"| {d['mode_balance_mean']:.2f} |")
    if not any_row:
        return "_No sim results yet. Run `scripts/eval_all.sh`._"
    return "\n".join(rows)


def real_table(runs_dir: Path) -> str:
    from grasprl.real.harness import load_results
    from grasprl.real.metrics import EvalMetrics

    header = "| arm | trials | success | grasp_slip | grabbed_nothing | \\|P(left)-0.5\\| |"
    rows = [header, "|---|---|---|---|---|---|"]
    any_row = False
    for label, name in ARMS:
        csv = runs_dir / f"real_{label}" / "eval" / "results.csv"
        results = load_results(csv)
        if not results:
            rows.append(f"| {name} | _not run_ | | | | |")
            continue
        any_row = True
        m = EvalMetrics(results)
        n = len(results)
        succ = sum(r.success for r in results) / n
        def rate(category: str, rs=results, total=n) -> float:
            return sum(1 for r in rs if r.failure_category
                       and r.failure_category.value == category) / total

        rows.append(f"| {name} | {n} | **{succ:.1%}** | {rate('grasp_slip'):.1%} "
                    f"| {rate('grabbed_nothing'):.1%} | {m.mode_balance_score():.2f} |")
    if not any_row:
        return "_No real-arm results yet. Run `grasprl.real.run` for each arm._"
    return "\n".join(rows)


def build(results_dir: str = "results", runs_dir: str = "runs",
          out: str = "results/comparison.md") -> str:
    results_dir, runs_dir = Path(results_dir), Path(runs_dir)
    cal = _load(results_dir / "calibration.json")
    critic = _load(Path(runs_dir, "gaf_critic", "summary.json"))
    ppo = _load(Path(runs_dir, "ppo_seed0", "summary.json"))

    parts = [
        "# Flow-SDE PPO vs Guided Action Flow, on SmolVLA grasp stability",
        "",
        "Both arms start from the **same frozen checkpoint** "
        "(`checkpoints/base_smolvla`), which scores 53% on the real SO-ARM101 and "
        "loses 30% of trials to grasp slip. They differ in where the improvement "
        "is allowed to live: PPO updates 99.8M policy parameters; Guided Action "
        "Flow updates none and steers the sampler with a learned action-chunk "
        "critic instead.",
        "",
        "## Sim (contact-physics grasp, held-out seeds)",
        "",
        sim_table(results_dir),
        "",
        "## Real SO-ARM101 (operator-scored, tier A)",
        "",
        real_table(runs_dir),
        "",
    ]
    if cal:
        b = cal["best"]
        parts += [
            "## Calibration",
            "",
            "Every sim number above is measured in a contact model tuned so the "
            "**frozen** policy fails the way it really does. This is the load-bearing "
            "assumption of the sim half of this comparison.",
            "",
            "| | success | grasp_slip | grabbed_nothing |",
            "|---|---|---|---|",
            f"| real arm (30 trials) | {cal['real_profile']['success']:.0%} "
            f"| {cal['real_profile']['grasp_slip']:.0%} "
            f"| {cal['real_profile']['grabbed_nothing']:.0%} |",
            f"| calibrated sim | {b['success']:.0%} | {b['slip']:.0%} "
            f"| {b['grabbed_nothing']:.0%} |",
            "",
            f"Chosen: `gripper_forcerange = {b['gripper_forcerange']}`, "
            f"`pad_friction[0] = {b['pad_friction']}` (L1 {b['distance']:.3f}, "
            f"gate {'passed' if cal['gate_passed'] else 'FAILED'}). "
            f"With a weld grasp, where slipping is impossible, the same policy "
            f"scores {cal['weld_sim_ceiling']:.0%} in sim -- so its perception and "
            f"positioning transfer, and the gap is grip.",
            "",
        ]
    if ppo:
        parts += ["## Arm A: Flow-SDE PPO", "",
                  f"- {ppo['updates']} updates, {ppo['env_steps']:,} control ticks, "
                  f"{ppo['episodes']} episodes, {ppo['wall_clock_s'] / 3600:.1f} h",
                  f"- final training success {ppo['final_success_rate']:.1%}, "
                  f"slip {ppo.get('final_slip_rate', float('nan')):.1%}",
                  "- Evaluated with the deterministic ODE sampler, not the SDE it "
                  "trained with: the exploration noise has a real cost and scoring it "
                  "would measure the exploration mechanism, not the learned policy.",
                  ""]
    if critic:
        c = critic["config"]
        parts += ["## Arm B: Guided Action Flow", "",
                  f"- critic: {c['depth']} x {c['hidden']} MLP, ensemble K={c['ensemble']}, "
                  f"{c['epochs']} epochs, features `{c['features']}`",
                  f"- trained on {critic['episodes']} frozen-policy episodes "
                  f"({critic['samples']:,} chunks), episode-level split",
                  f"- final val MSE {critic['final_val_mse']:.5f}, "
                  f"ensemble disagreement {critic['final_ensemble_std']:.4f}",
                  "- Guidance hyperparameters were chosen on the **validation** seed "
                  "range and reported once on the disjoint held-out range.",
                  ""]
    text = "\n".join(parts)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    print(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="results/comparison.md")
    a = p.parse_args(argv)
    build(a.results_dir, a.runs_dir, a.out)


if __name__ == "__main__":
    main()
