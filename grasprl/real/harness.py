"""Evaluation harness: three-tier rollout runner and failure logger.

Ported from the sibling diffusion-policy repo
(``diffusion_policy_soarm/eval/harness.py``) so SmolVLA is scored with the same
protocol and the same CSV schema, keeping the two methods directly comparable.

Runs episodic rollouts on the real SO-ARM101 and records one row per trial:
- Tier A: cube positions from the training distribution.
- Tier B: cube positions shifted outside training range.
- Tier C: training positions + distractor objects.
- For each trial: binary success, failure category (first-cause priority),
  episode duration, and -- for successes -- which cube was chosen (left/right),
  so ``metrics.py`` can report the mode-balance score.
- Layouts are placed by the operator (approximate, not exact coordinates).

There is no simulator: every trial is a physical rollout scored by a human via
the interactive prompt after each episode.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from lerobot.robots.so_follower.so_follower import SOFollower

from grasprl.real.infer import StopReason, move_to_init, run_episode


class FailureCategory(str, Enum):
    GRABBED_NOTHING = "grabbed_nothing"
    GRASPED_WRONG_OBJECT = "grasped_wrong_object"
    GRASP_SLIP = "grasp_slip"
    MISSED_CUP = "missed_cup"
    KNOCKED_CUP_OVER = "knocked_cup_over"
    COLLISION_UNSAFE = "collision_unsafe"
    FROZE_NO_ATTEMPT = "froze_no_attempt"
    TIMEOUT = "timeout"


@dataclass
class TrialResult:
    """Result record for one evaluation rollout."""

    tier: str
    trial_idx: int
    success: bool
    failure_category: FailureCategory | None = None
    # "left" or "right" for successes, None otherwise.
    cube_chosen: str | None = None
    # Wall-clock episode duration in seconds.
    duration_s: float = 0.0
    notes: str = ""


# CSV column order. Kept identical to the diffusion repo so the metrics loader
# (and any cross-repo comparison) agrees.
CSV_FIELDS = [
    "tier", "trial_idx", "success", "failure_category", "cube_chosen", "duration_s", "notes",
]


def _row_from_result(r: TrialResult) -> dict:
    """Flatten a TrialResult into CSV-friendly primitives."""
    d = asdict(r)
    d["success"] = int(r.success)
    d["failure_category"] = r.failure_category.value if r.failure_category else ""
    d["cube_chosen"] = r.cube_chosen or ""
    return d


def _result_from_row(row: dict) -> TrialResult:
    """Parse one CSV row back into a TrialResult."""
    cat = row["failure_category"].strip()
    cube = row["cube_chosen"].strip()
    return TrialResult(
        tier=row["tier"],
        trial_idx=int(row["trial_idx"]),
        success=bool(int(row["success"])),
        failure_category=FailureCategory(cat) if cat else None,
        cube_chosen=cube or None,
        duration_s=float(row["duration_s"]) if row.get("duration_s") else 0.0,
        notes=row.get("notes", ""),
    )


def load_results(csv_path: Path) -> list[TrialResult]:
    """Load all TrialResults from a results CSV (empty list if absent)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return [_result_from_row(row) for row in csv.DictReader(f)]


def _prompt_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    """Show a numbered menu and return the chosen option string."""
    while True:
        print(prompt)
        for i, opt in enumerate(options, 1):
            marker = " (default)" if opt == default else ""
            print(f"  {i}. {opt}{marker}")
        raw = input("> ").strip()
        if raw == "" and default is not None:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid choice, try again.")


def _prompt_yes_no(prompt: str) -> bool:
    while True:
        raw = input(f"{prompt} [y/n] ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


class EvalHarness:
    """Runs tiered evaluation trials on the arm and accumulates results.

    The control loop and arm motion come from :mod:`grasprl.real.infer`
    (``run_episode``, ``move_to_init``); this class adds the trial loop, the
    operator scoring prompt, and CSV logging.

    It takes an :class:`~grasprl.policy.actor.Actor` rather than a bare policy,
    which is what lets all three arms of the comparison -- including Guided
    Action Flow, whose critic intervenes inside the denoising loop and so cannot
    be expressed as ``policy.select_action`` -- run through one identical loop.
    """

    def __init__(
        self,
        actor,
        robot: SOFollower,
        motor_names: list[str],
        init_pose: np.ndarray,
        fps: float,
        max_steps: int,
        dump_frames_dir: Path | None = None,
    ) -> None:
        self.actor = actor
        self.robot = robot
        self.motor_names = motor_names
        self.init_pose = init_pose
        self.fps = float(fps)
        self.max_steps = int(max_steps)
        self.dump_frames_dir = dump_frames_dir

    def _score_trial(
        self, tier: str, trial_idx: int, stop_reason: StopReason, duration_s: float
    ) -> TrialResult:
        """Interactively record the operator's judgement of one rollout."""
        print(
            f"\n--- score trial {trial_idx} (tier {tier}, "
            f"{duration_s:.1f} s, stop={stop_reason.value}) ---"
        )
        success = _prompt_yes_no("Success? (cube released inside the blue cup)")

        failure_category: FailureCategory | None = None
        cube_chosen: str | None = None
        if success:
            cube_chosen = _prompt_choice("Which cube did it pick?", ["left", "right"])
        else:
            # Default to "timeout" when the episode hit the step cap.
            default = FailureCategory.TIMEOUT.value if stop_reason == StopReason.TIMEOUT else None
            cat = _prompt_choice(
                "Failure category (first cause):",
                [c.value for c in FailureCategory],
                default=default,
            )
            failure_category = FailureCategory(cat)

        notes = input("Notes (optional): ").strip()
        return TrialResult(
            tier=tier,
            trial_idx=trial_idx,
            success=success,
            failure_category=failure_category,
            cube_chosen=cube_chosen,
            duration_s=duration_s,
            notes=notes,
        )

    def run_tier(
        self, tier: str, num_trials: int, output_dir: Path, start_idx: int = 0
    ) -> list[TrialResult]:
        """Execute rollouts ``start_idx..num_trials`` for the given tier.

        Each scored trial is appended to ``output_dir/results.csv`` immediately,
        so an interrupted session loses no completed trials and can resume.
        """
        results: list[TrialResult] = []
        for trial_idx in range(start_idx, num_trials):
            input(
                f"\n=== Tier {tier}, trial {trial_idx + 1}/{num_trials} ===\n"
                f"Place the cubes for tier {tier}, clear the workspace, then press Enter to start. "
                f"During the episode press Enter to end it (task done or clearly failed); "
                f"it auto-stops after {self.max_steps} steps."
            )
            move_to_init(self.robot, self.motor_names, self.init_pose, self.fps)

            # Dump camera frames only on the first trial of the session.
            dump_dir = self.dump_frames_dir if trial_idx == start_idx else None
            self.actor.reset(trial_idx)
            stop_reason, duration_s = run_episode(
                self.actor, self.robot, self.motor_names, self.fps,
                max_steps=self.max_steps,
                dump_frames_dir=dump_dir,
            )

            # Auto-return so only the cubes need re-placing for the next trial.
            move_to_init(self.robot, self.motor_names, self.init_pose, self.fps)

            result = self._score_trial(tier, trial_idx, stop_reason, duration_s)
            self.save_results([result], output_dir)
            results.append(result)
        return results

    def save_results(self, results: list[TrialResult], output_dir: Path) -> Path:
        """Append results to ``output_dir/results.csv`` (creating header once)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "results.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow(_row_from_result(r))
        return csv_path
