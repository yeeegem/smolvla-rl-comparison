"""Operator-scored evaluation of one arm of the comparison on the real SO-ARM101.

Runs ``--trials`` rollouts of ``--method`` for one tier, prompts the operator to
score each trial, and appends to a CSV whose schema is byte-identical to
``sim2real-soarm-benchmark``'s -- so the numbers here mean exactly what the 53%
success / 30% grasp-slip baseline means, and drop straight into the same table.

All three arms run through the same :class:`grasprl.policy.actor.Actor` and the
same control loop, at the same ``--n-exec`` decision cadence used in sim::

    # the frozen policy, re-anchored inside this run rather than compared to a
    # historical number measured at a different decision cadence
    uv run python -m grasprl.real.run --method base \
        --checkpoint checkpoints/base_smolvla --tier A --trials 20

    # Arm A: PPO-updated weights
    uv run python -m grasprl.real.run --method ppo \
        --checkpoint runs/ppo_seed0/checkpoints/last/pretrained_model --tier A --trials 20

    # Arm B: the SAME frozen policy, steered by the critic
    uv run python -m grasprl.real.run --method gaf \
        --checkpoint checkpoints/base_smolvla --critic runs/gaf_critic \
        --tier A --trials 20

Aggregate with ``python -m grasprl.real.metrics <results.csv>``.
On-arm runs are executed by the operator, not in CI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from grasprl.envs.pickplace_env import DEFAULT_TASK
from grasprl.real.harness import EvalHarness, load_results
from grasprl.real.infer import build_real_actor, connect_robot, read_pose

DEFAULT_CHECKPOINT = "checkpoints/base_smolvla"
DEFAULT_CONFIG = "configs/eval_real.yaml"


def _set_override(cfg: dict, dotted_key: str, value: str) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load the YAML eval config and apply ``key.nested=value`` CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        key, value = ov.split("=", 1)
        _set_override(cfg, key, value)
    return cfg


def resolve_task(cfg: dict) -> str:
    """The language instruction the policy was trained with.

    Defaults to the constant the sim env uses, so sim and real feed the policy
    the identical string; ``eval.task`` in the config can override it.
    """
    return str(cfg.get("eval", {}).get("task") or DEFAULT_TASK)


def resolve_init_pose(cfg: dict, robot, motor_names: list[str]) -> np.ndarray:
    configured = cfg.get("infer", {}).get("init_pose")
    if configured is not None:
        return np.asarray(list(configured), dtype=np.float32)
    print("infer.init_pose is null: capturing the arm's current pose as the init pose.")
    return read_pose(robot, motor_names)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Operator-scored real-arm evaluation of one comparison arm.")
    parser.add_argument("--method", default="base", choices=["base", "ppo", "gaf"],
                        help="which arm of the comparison to score")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="SmolVLA pretrained_model/ dir. For --method gaf this must "
                             "be the BASE checkpoint: the method's whole claim is that "
                             "the policy is unchanged.")
    parser.add_argument("--critic", default=None,
                        help="trained critic dir, required for --method gaf")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to eval config YAML.")
    parser.add_argument("--tier", default="A", choices=["A", "B", "C"],
                        help="Scene setup to score (default A: cubes in the trained layout).")
    parser.add_argument("--trials", type=int, default=None,
                        help="Trials for this tier (default: eval.num_trials_per_tier).")
    parser.add_argument("--n-exec", type=int, default=10,
                        help="actions executed per decision; must match the sim runs")
    parser.add_argument("--output", default=None,
                        help="Output dir for results.csv (default: runs/real_<method>/eval).")
    parser.add_argument("--debug-frames", action="store_true",
                        help="Dump the first trial's camera frames (as the policy sees them).")
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, overrides or None)
    num_trials = args.trials if args.trials is not None else int(cfg["eval"]["num_trials_per_tier"])
    output_dir = Path(args.output) if args.output else Path("runs") / f"real_{args.method}" / "eval"

    existing = [r for r in load_results(output_dir / "results.csv") if r.tier == args.tier]
    start_idx = len(existing)
    if start_idx >= num_trials:
        print(f"Tier {args.tier} already has {start_idx}/{num_trials} trials. Nothing to do.")
        return
    if start_idx > 0:
        print(f"Resuming tier {args.tier} at trial {start_idx + 1}/{num_trials}.")

    guidance = None
    if args.method == "gaf":
        from grasprl.gaf.guided_sampler import GuidanceConfig
        raw = yaml.safe_load(Path("configs/gaf.yaml").read_text()).get("guidance", {})
        guidance = GuidanceConfig(**raw)
        print(f"Guidance: {guidance}")

    task = resolve_task(cfg)
    print(f"Task instruction: {task!r}")
    actor = build_real_actor(args.method, args.checkpoint, args.critic, task,
                             n_exec=args.n_exec, guidance=guidance)
    robot, motor_names = connect_robot(cfg)

    try:
        init_pose = resolve_init_pose(cfg, robot, motor_names)
        harness = EvalHarness(
            actor, robot, motor_names, init_pose,
            fps=float(cfg["dataset"]["fps"]),
            max_steps=int(cfg["eval"]["max_episode_steps"]),
            dump_frames_dir=(output_dir / "debug_frames") if args.debug_frames else None,
        )
        results = harness.run_tier(args.tier, num_trials, output_dir, start_idx=start_idx)
        print(f"\nSaved {len(results)} trial(s) to {output_dir / 'results.csv'}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
