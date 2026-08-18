"""Evaluation metrics: success rate, failure breakdown, mode-balance score.

Ported from the sibling diffusion-policy repo
(``diffusion_policy_soarm/eval/metrics.py``); the metric definitions are
identical so numbers are directly comparable across the sibling repos, and only
the import path and the CLI entry point below differ.
Aggregates the per-trial rows written by ``EvalHarness`` into the numbers and
Markdown tables used in the README and the writeup:
- Success rate per tier.
- Failure category distribution (first-cause priority breakdown), as a share of
  the failed trials in the tier (sums to 100%).
- Mode-balance score: |P(left) - 0.5| among successes -- lower is better, 0
  means a perfect 50/50 split matching the training data.
- Mean wall-clock duration of successful episodes, per tier.

Aggregate a run's CSV into tables.md / metrics.json with::

    uv run python -m grasprl.real.metrics runs/real_ppo/eval/results.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from grasprl.real.harness import TrialResult, load_results


def _pct(x: float) -> str:
    """Format a fraction as a percentage, or 'n/a' for NaN."""
    return "n/a" if math.isnan(x) else f"{x * 100:.1f}%"


def _num(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.1f}"


class EvalMetrics:
    """Aggregate and format evaluation metrics for one set of trial results."""

    def __init__(self, results: list[TrialResult]) -> None:
        self.results = list(results)

    @classmethod
    def from_csv(cls, csv_path: Path) -> EvalMetrics:
        return cls(load_results(csv_path))

    # -- selectors ----------------------------------------------------------

    def tiers(self) -> list[str]:
        """Tiers present in the results, in A/B/C order."""
        present = {r.tier for r in self.results}
        return [t for t in ("A", "B", "C") if t in present] + sorted(present - {"A", "B", "C"})

    def _tier(self, tier: str) -> list[TrialResult]:
        return [r for r in self.results if r.tier == tier]

    # -- metrics ------------------------------------------------------------

    def success_rate(self, tier: str) -> float:
        """Fraction of successful trials for the given tier (NaN if none)."""
        trials = self._tier(tier)
        if not trials:
            return math.nan
        return sum(r.success for r in trials) / len(trials)

    def failure_breakdown(self, tier: str) -> dict[str, float]:
        """Per-category share of the failed trials in the tier.

        The denominator is the number of categorised failures, not the number of
        trials, so the fractions sum to 1 whatever the success rate is. A failed
        trial with no category recorded is excluded from both sides of the ratio.
        """
        counts: dict[str, int] = {}
        for r in self._tier(tier):
            if not r.success and r.failure_category is not None:
                counts[r.failure_category.value] = counts.get(r.failure_category.value, 0) + 1
        n_failures = sum(counts.values())
        if not n_failures:
            return {}
        return {cat: c / n_failures for cat, c in counts.items()}

    def top_failures(self, tier: str, k: int | None = None) -> list[tuple[str, float]]:
        """Observed failure categories in the tier, most common first.

        Categories with no occurrences are absent rather than listed at 0%. With
        ``k`` set, only the k most common are returned; by default all of them
        are, so the shares sum to 100%.
        """
        items = sorted(self.failure_breakdown(tier).items(), key=lambda kv: kv[1], reverse=True)
        return items if k is None else items[:k]

    def mode_balance_score(self) -> float:
        """|P(left) - 0.5| among trials that recorded a cube choice (NaN if none)."""
        chosen = [r.cube_chosen for r in self.results if r.cube_chosen in ("left", "right")]
        if not chosen:
            return math.nan
        p_left = chosen.count("left") / len(chosen)
        return abs(p_left - 0.5)

    def mode_split(self) -> tuple[float, float]:
        """(P(left), P(right)) among recorded cube choices; (NaN, NaN) if none."""
        chosen = [r.cube_chosen for r in self.results if r.cube_chosen in ("left", "right")]
        if not chosen:
            return math.nan, math.nan
        p_left = chosen.count("left") / len(chosen)
        return p_left, 1.0 - p_left

    def mean_success_duration_s(self, tier: str) -> float:
        """Mean wall-clock duration of successful episodes in the tier (NaN if none)."""
        durations = [r.duration_s for r in self._tier(tier) if r.success]
        if not durations:
            return math.nan
        return sum(durations) / len(durations)

    # -- rendering ----------------------------------------------------------

    def to_markdown(self) -> str:
        """Self-contained Markdown summary: per-tier table + failure breakdown."""
        lines: list[str] = []
        lines.append("### Per-tier summary\n")
        lines.append("| Tier | Trials | Success rate | Mean success time (s) |")
        lines.append("|---|---|---|---|")
        for tier in self.tiers():
            n = len(self._tier(tier))
            lines.append(
                f"| {tier} | {n} | {_pct(self.success_rate(tier))} "
                f"| {_num(self.mean_success_duration_s(tier))} |"
            )

        p_left, p_right = self.mode_split()
        score = self.mode_balance_score()
        score_str = "n/a" if math.isnan(score) else f"{score:.3f}"
        lines.append("")
        lines.append(
            f"Mode balance: left {_pct(p_left)} / right {_pct(p_right)}, "
            f"|P(left) - 0.5| = {score_str}"
        )

        lines.append("\n### Failure breakdown (share of failed trials)\n")
        lines.append("| Tier | Category | Share of failures |")
        lines.append("|---|---|---|")
        for tier in self.tiers():
            breakdown = sorted(
                self.failure_breakdown(tier).items(), key=lambda kv: kv[1], reverse=True
            )
            if not breakdown:
                lines.append(f"| {tier} | (none) | - |")
            for cat, frac in breakdown:
                lines.append(f"| {tier} | {cat} | {_pct(frac)} |")

        return "\n".join(lines) + "\n"

    def save(self, output_dir: Path) -> None:
        """Write metrics.json and tables.md to ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode_balance_score": self.mode_balance_score(),
            "tiers": {
                tier: {
                    "trials": len(self._tier(tier)),
                    "success_rate": self.success_rate(tier),
                    "mean_success_duration_s": self.mean_success_duration_s(tier),
                    "failure_breakdown": self.failure_breakdown(tier),
                }
                for tier in self.tiers()
            },
        }
        (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "tables.md").write_text(self.to_markdown())


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate SmolVLA eval results.csv.")
    parser.add_argument("results_csv", help="Path to results.csv written by the harness.")
    parser.add_argument(
        "--output", default=None,
        help="Dir for metrics.json / tables.md (default: results.csv's dir).",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.results_csv)
    metrics = EvalMetrics.from_csv(csv_path)
    output_dir = Path(args.output) if args.output else csv_path.parent
    metrics.save(output_dir)
    print(metrics.to_markdown())
    print(f"Wrote {output_dir / 'metrics.json'} and {output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
