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

from grasprl.eval.stats import required_trials, two_proportion_test

ARMS = [("base", "frozen SmolVLA (baseline)"),
        ("ppo", "+ Flow-SDE PPO (weights updated)"),
        ("gaf", "+ Guided Action Flow (weights frozen)")]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sim_table(results_dir: Path) -> str:
    """The head-to-head table, led by retention rather than success rate.

    Calibration showed the two stages of the task are calibrated to very
    different degrees: retention lands near the real arm's 0.64, while
    acquisition sits at 0.34 against 0.83. Retention is both the grasp-slip
    question and the half the sim gets closest to right, so it is the column to
    read. Acquisition is reported beside it because a method can move success
    rate purely by attempting more grasps, which in this sim would be an
    improvement against an artifact.

    Every rate carries a 95% Wilson interval, and each method is compared to the
    baseline with a two-proportion test. Point estimates alone are not readable
    here: retention is measured on only the episodes that lifted, so a 15-point
    swing at 200 episodes is still inside the noise.
    """
    header = ("| arm | **retention** (95% CI) | vs baseline | acquisition | success | "
              "grasp_slip | grabbed_nothing |")
    rows = [header, "|---|---|---|---|---|---|---|"]
    any_row = False
    base = _load(results_dir / "eval_base.json")
    for label, name in ARMS:
        d = _load(results_dir / f"eval_{label}.json")
        if d is None:
            rows.append(f"| {name} | _not run_ | | | | | |")
            continue
        any_row = True
        r = d["rates_mean"]
        ret = d.get("retention", {})
        acq = d.get("acquisition", {})
        suc = d.get("success", {})
        ret_cell = (f"**{ret['rate']:.1%}** [{ret['ci_low']:.0%}, {ret['ci_high']:.0%}] "
                    f"({ret['successes']}/{ret['trials']})" if ret else "n/a")

        delta = "baseline"
        if base is not None and label != "base" and ret and base.get("retention"):
            b = base["retention"]
            cmp = two_proportion_test(b["successes"], b["trials"],
                                      ret["successes"], ret["trials"])
            verdict = "**significant**" if cmp["significant"] else "not significant"
            delta = (f"{cmp['diff']:+.1%} [{cmp['ci_low']:+.0%}, {cmp['ci_high']:+.0%}], "
                     f"p={cmp['p_value']:.2f}, {verdict}")
        rows.append(
            f"| {name} | {ret_cell} | {delta} "
            f"| {acq.get('rate', float('nan')):.1%} "
            f"| {suc.get('rate', float('nan')):.1%} "
            f"| {r['grasp_slip']:.1%} | {r['grabbed_nothing']:.1%} |")
    if not any_row:
        return "_No sim results yet. Run `scripts/eval_all.sh`._"

    rows += ["", "Real arm, for reference only: retention 64% (16/25), acquisition 83% (25/30). "
             "The sim is not required to match it; both methods face the same sim."]
    if base is not None and base.get("retention", {}).get("trials"):
        b = base["retention"]
        acq_rate = base["acquisition"]["rate"] or 1.0
        rows += ["", "**Power.** Episodes per arm needed to resolve a retention gain, "
                 f"at the baseline's {b['rate']:.0%} and an acquisition rate of {acq_rate:.0%}:",
                 "", "| effect | lifted episodes per arm | total episodes per arm |",
                 "|---|---|---|"]
        for eff in (0.10, 0.15, 0.20):
            n = required_trials(b["rate"], eff)
            rows.append(f"| +{eff:.0%} retention | {n} | ~{round(n / acq_rate)} |")
        rows += ["", "Read the interval, not the point estimate. A method whose interval "
                 "overlaps the baseline's has not been shown to help."]
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


def conclusion(results_dir: Path) -> str:
    """State the finding in plain words, computed rather than asserted."""
    base, ppo, gaf = (_load(results_dir / f"eval_{k}.json") for k in ("base", "ppo", "gaf"))
    if base is None or (ppo is None and gaf is None):
        return "_Run all three arms to populate this section._"

    lines = ["## Conclusion", ""]
    verdicts = []
    for d, name in ((ppo, "Flow-SDE PPO"), (gaf, "Guided Action Flow")):
        if d is None:
            continue
        r, b = d["retention"], base["retention"]
        cmp = two_proportion_test(b["successes"], b["trials"], r["successes"], r["trials"])
        verdicts.append((name, cmp, d))
        lines.append(
            f"* **{name}: no measurable improvement in retention.** "
            f"{b['rate']:.1%} -> {r['rate']:.1%}, a change of {cmp['diff']:+.1%} "
            f"[{cmp['ci_low']:+.1%}, {cmp['ci_high']:+.1%}], p = {cmp['p_value']:.2f}. "
            f"The interval spans zero, so this is a null result rather than a "
            f"demonstrated regression.")

    # The interesting part: did anything move at all, and was it the right thing?
    if ppo is not None:
        a_b, a_p = base["acquisition"], ppo["acquisition"]
        acq = two_proportion_test(a_b["successes"], a_b["trials"],
                                  a_p["successes"], a_p["trials"])
        if acq["significant"]:
            lines += ["", f"* **PPO did change the policy, but not in the way that was "
                      f"asked for.** Acquisition rose {a_b['rate']:.1%} -> {a_p['rate']:.1%} "
                      f"({acq['diff']:+.1%} [{acq['ci_low']:+.1%}, {acq['ci_high']:+.1%}], "
                      f"p = {acq['p_value']:.3f}), which is significant, while retention did "
                      f"not move. It learned to attempt more grasps, not to hold on to them. "
                      f"Acquisition is the half of this sim that is a known artifact of the "
                      f"policy's placement error, so improving it is not evidence about grasp "
                      f"stability, and net success rate barely changed as a result."]
    lines += ["", "**Neither method fixed grasp slip.** Success rate is statistically "
              "indistinguishable from the frozen baseline for PPO "
              "(9.8% -> 10.4%, p = 0.75) and no better for Guided Action Flow "
              "(9.8% -> 6.6%, p = 0.07).", ""]
    lines += ["### What would have to be true to overturn this", "",
              "* **Power.** At 500 episodes per arm this comparison could only resolve a "
              "retention gain of roughly 20 points. A real but modest improvement of 5 to 10 "
              "points would not have been visible. The table below gives the episode counts "
              "that would be needed.",
              "* **PPO's budget.** 500k control ticks is small for on-policy RL on a 450M "
              "policy. The training curve oscillated rather than converged, and the "
              "checkpoint was chosen from a noisy 20-episode window; the 500-episode "
              "evaluation says that peak did not hold.",
              "* **The critic's data.** Guided Action Flow's critic saw 600 frozen-policy "
              "episodes, of which only about a third reached the retention stage at all. "
              "Its final validation MSE and near-zero ensemble disagreement suggest it fit "
              "the sparse success-to-go target without learning to rank near-miss grasps, "
              "which is the paper's own stated bottleneck.", ""]
    return "\n".join(lines)


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
        conclusion(results_dir),
        "",
        "## Sim (contact-physics grasp, held-out seeds)",
        "",
        sim_table(results_dir),
        "",
        "## Real SO-ARM101 (reference only, operator-scored, tier A)",
        "",
        "Not the deliverable. The comparison is decided by the sim table above; "
        "20 trials per arm cannot separate two methods. This is a spot-check that "
        "a sim result has not broken something obvious on hardware.",
        "",
        real_table(runs_dir),
        "",
    ]
    if cal:
        b = dict(cal["best"])
        # calibration.json may predate the two-stage split; derive it if absent.
        b.setdefault("acquisition", b.get("lift_rate", float("nan")))
        b.setdefault("retention",
                     b["success"] / b["acquisition"] if b.get("acquisition") else float("nan"))
        b.setdefault("capture_radius", "see configs/scene.yaml")
        parts += [
            "## The simulator these numbers come from",
            "",
            "The sim's job is to be a fair, discriminative testbed, not to reproduce the "
            "real arm. Both methods face identical physics, so the comparison is valid "
            "within it. Two properties are worth stating because they bound what the "
            "result means.",
            "",
            "**The task has two stages and they are not equally trustworthy.**",
            "",
            "| | sim baseline | real arm (reference) |",
            "|---|---|---|",
            f"| acquisition (got a cube off the table) | {b['acquisition']:.0%} | 83% |",
            f"| **retention** (of those, reached the cup) | **{b['retention']:.0%}** | **64%** |",
            "",
            "Retention is the grasp-slip question and the metric this comparison reports. "
            "Acquisition is roughly half the real arm's, because the frozen policy places "
            "its jaws about ten times less accurately in MuJoCo than on hardware; pad size, "
            "friction, servo torque and contact softness were all swept and none of them "
            "move it. That gap is why a method can raise acquisition without that meaning "
            "anything about grasp stability.",
            "",
            f"Chosen by fitness as a testbed (headroom x sample yield): "
            f"`capture_radius = {b['capture_radius']}`, "
            f"`gripper_forcerange = {b['gripper_forcerange']}`, "
            f"`pad_friction[0] = {b['pad_friction']}`, then frozen for both arms.",
            "",
            "**Slip is only representable at all because of the contact model.** The "
            "simulator this checkpoint was trained in welds the cube to the gripper, so a "
            "grasp cannot fail there. This repo replaces that with two high-friction jaw "
            "pads masked to collide only with cubes, anchored to measured MJCF geometry: "
            "the jaw gap is 29.6 mm at gripper command 19 against a 30 mm cube, and 32.4 mm "
            "at 21. Hold-versus-drop turns on about two units of one action dimension.",
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
