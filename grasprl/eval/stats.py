"""Confidence intervals and two-proportion tests for the comparison.

Every headline number here is a proportion measured over i.i.d. episodes, so its
uncertainty is available analytically and does not need to be estimated by
splitting the run into groups. Averaging rates over a handful of seeds and
reporting their spread throws away most of the information: with two seeds the
spread is a one-degree-of-freedom variance estimate, which is far noisier than
the binomial interval you can compute from the counts directly.

So runs pool **counts**, not rates, and report Wilson score intervals.

Retention needs this most. It is measured only over the episodes that got a cube
off the table, so at an acquisition rate of ~0.34 a 200-episode run yields fewer
than 70 retention samples, and the interval on it is roughly three times wider
than the one on success rate. Reporting it without an interval invites reading
noise as an effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 1.96 = two-sided 95%.
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Proportion:
    """A measured rate with its Wilson score interval."""

    successes: int
    trials: int
    lo: float
    hi: float

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else float("nan")

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2

    def __str__(self) -> str:
        if not self.trials:
            return "n/a"
        return f"{self.rate:.1%} [{self.lo:.1%}, {self.hi:.1%}] (n={self.trials})"

    def to_dict(self) -> dict:
        return {"rate": self.rate, "successes": self.successes, "trials": self.trials,
                "ci_low": self.lo, "ci_high": self.hi}


def wilson(successes: int, trials: int, z: float = Z95) -> Proportion:
    """Wilson score interval.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly at small ``trials`` and at rates near 0 or 1 -- both of
    which this project hits, since retention is measured on a third of episodes
    and some cells grasp almost nothing.
    """
    if trials <= 0:
        return Proportion(0, 0, float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return Proportion(successes, trials, max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Compare two rates: difference, its 95% interval, and a two-sided p-value.

    This is the test that answers "did this method actually beat the baseline",
    as opposed to "is its point estimate higher". The interval on the difference
    is the thing to report; the p-value is a convenience.
    """
    if n1 <= 0 or n2 <= 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "significant": False}
    p1, p2 = k1 / n1, k2 / n2
    diff = p2 - p1

    # Interval on the difference: unpooled standard error (Wald), which is the
    # standard choice for a difference of independent proportions.
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    lo, hi = diff - Z95 * se, diff + Z95 * se

    # Test: pooled standard error under the null that the rates are equal.
    p_pool = (k1 + k2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"diff": diff, "ci_low": lo, "ci_high": hi, "z": z,
            "p_value": p_value, "significant": bool(p_value < 0.05)}


def required_trials(baseline: float, effect: float, power: float = 0.80) -> int:
    """Episodes per arm needed to resolve ``effect`` at 95%/``power``.

    Used to size runs honestly rather than discovering afterwards that the
    comparison could never have separated the two methods. Note the answer is in
    *retention samples*; divide by the acquisition rate to get episodes.
    """
    z_a, z_b = Z95, {0.80: 0.8416, 0.90: 1.2816}.get(power, 0.8416)
    p1, p2 = baseline, min(max(baseline + effect, 1e-6), 1 - 1e-6)
    p_bar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)
