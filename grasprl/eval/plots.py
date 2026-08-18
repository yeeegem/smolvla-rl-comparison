"""Curves for the report: PPO's learning trace and the GAF guidance sweep.

Matplotlib only -- no wandb, no tensorboard. The runs write JSONL/JSON and these
read it, so a figure can always be regenerated from what is on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_metrics(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _smooth(y: np.ndarray, window: int = 15) -> np.ndarray:
    if len(y) < window or window < 2:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def learning_curve(metrics_path: str | Path, out: str | Path,
                   baseline: float | None = None) -> Path:
    """PPO success and slip rate against environment steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_metrics(metrics_path)
    steps = np.array([r["env_steps"] for r in rows], float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for a, key, title in ((ax[0], "success_rate", "success rate"),
                          (ax[1], "slip_rate", "grasp-slip rate")):
        y = np.array([r.get(key, np.nan) for r in rows], float)
        a.plot(steps, y, alpha=0.25, color="tab:blue")
        s = _smooth(y)
        a.plot(steps[len(steps) - len(s):], s, color="tab:blue", lw=2)
        if baseline is not None and key == "success_rate":
            a.axhline(baseline, ls="--", color="gray", label="frozen baseline")
            a.legend()
        a.set_xlabel("control ticks")
        a.set_title(title)
        a.grid(alpha=0.3)
    fig.suptitle("Flow-SDE PPO")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return Path(out)


def guidance_sweep(sweep_path: str | Path, out: str | Path) -> Path:
    """Validation success against guidance strength.

    ``beta`` divides the gradient, so the x-axis runs from strong (left) to
    gentle (right); the baseline line is what "no guidance at all" scores on the
    same seeds, which is the only thing that makes a point on this curve mean
    anything.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(Path(sweep_path).read_text())
    rows = report["grid"]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for alpha in sorted({r["alpha"] for r in rows}):
        pts = sorted([r for r in rows if r["alpha"] == alpha], key=lambda r: r["beta"])
        ax.plot([r["beta"] for r in pts], [r["success"] for r in pts],
                marker="o", label=f"alpha={alpha}")
    ax.axhline(report["baseline"]["success"], ls="--", color="gray", label="unguided")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("beta  (smaller = stronger guidance)")
    ax.set_ylabel("validation success rate")
    ax.set_title("Guided Action Flow: guidance strength")
    ax.grid(alpha=0.3)
    ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return Path(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve")
    c.add_argument("--metrics", default="runs/ppo_seed0/metrics.jsonl")
    c.add_argument("--out", default="results/ppo_learning_curve.png")
    c.add_argument("--baseline", type=float, default=None)
    s = sub.add_parser("sweep")
    s.add_argument("--sweep", default="results/gaf_sweep.json")
    s.add_argument("--out", default="results/gaf_sweep.png")
    a = p.parse_args(argv)
    if a.cmd == "curve":
        print(learning_curve(a.metrics, a.out, a.baseline))
    else:
        print(guidance_sweep(a.sweep, a.out))


if __name__ == "__main__":
    main()
