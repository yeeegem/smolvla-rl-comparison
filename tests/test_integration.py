"""End-to-end smoke tests: the pipeline stages actually run and hand off.

These are deliberately tiny (a handful of episodes, a two-epoch critic) -- they
check that the pieces connect, not that anything learns. The expensive runs are
the operator's; what CI can usefully catch is a shape mismatch between
collection and guidance, or an evaluator that silently scores the wrong arm.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

gpu_only = pytest.mark.skipif(not torch.cuda.is_available(),
                              reason="SmolVLA needs a GPU")

CHECKPOINT = "checkpoints/base_smolvla"


def _have_checkpoint() -> bool:
    from pathlib import Path
    return (Path(CHECKPOINT) / "config.json").exists()


needs_checkpoint = pytest.mark.skipif(
    not _have_checkpoint(),
    reason="run scripts/fetch_base_checkpoint.sh first")


def test_env_step_advances_exactly_n_exec_ticks():
    from grasprl.envs.pickplace_env import EnvConfig, PickPlaceEnv

    env = PickPlaceEnv(cfg=EnvConfig(n_exec=7, domain_randomize=False))
    try:
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"front", "wrist", "state"}
        assert obs["state"].shape == (6,) and obs["state"].dtype == np.float32
        action = np.tile(obs["state"], (7, 1))
        _obs, _r, _t, _tr, info = env.step(action)
        assert info["step_ticks"] == 7
        assert env._ticks == 7
    finally:
        env.close()


def test_actions_are_clipped_to_the_calibrated_range():
    from grasprl.envs.pickplace_env import EnvConfig, PickPlaceEnv

    env = PickPlaceEnv(cfg=EnvConfig(n_exec=2, domain_randomize=False))
    try:
        env.reset(seed=0)
        env.step(np.full((2, 6), 1e4, dtype=np.float32))
        state = env.scene.get_state()
        assert (state[:5] <= env.action_high[0, :5] + 1.0).all()
        assert 0.0 <= state[5] <= 100.0
    finally:
        env.close()


def test_vec_env_autoresets_and_reports_the_final_episode():
    from grasprl.envs.pickplace_env import EnvConfig
    from grasprl.envs.vec_env import VecPickPlaceEnv

    envs = VecPickPlaceEnv(2, cfg=EnvConfig(n_exec=10, max_ticks=20,
                                            domain_randomize=False), seed=0)
    try:
        obs = envs.reset(seed=0)
        seen = None
        for _ in range(4):
            actions = np.stack([np.tile(o["state"], (10, 1)) for o in obs])
            obs, _r, dones, infos = envs.step(actions)
            for i, d in enumerate(dones):
                if d:
                    seen = infos[i]["final_info"]
        assert seen is not None, "max_ticks=20 must truncate within 4 decisions"
        assert set(seen) >= {"success", "category", "slips", "ticks"}
    finally:
        envs.close()


@gpu_only
@needs_checkpoint
def test_gaf_pipeline_collect_train_guide(tmp_path):
    """Collection -> critic -> guided sampling, with the shapes lining up.

    The failure this catches is a silent one: if collection stores chunks in a
    different space or layout than guidance differentiates, everything still
    runs and the guidance is meaningless.
    """
    from grasprl.envs.pickplace_env import DEFAULT_TASK
    from grasprl.gaf.collect import collect
    from grasprl.gaf.train_critic import train
    from grasprl.policy.actor import build_actor

    roll = tmp_path / "rollouts"
    meta = collect(CHECKPOINT, str(roll), episodes=2, n_envs=2, n_exec=10,
                   max_ticks=60, domain_randomize=False, seed=0)
    assert meta["episodes"] == 2 and meta["samples"] > 0

    data = np.load(roll / "rollouts.npz")
    assert data["action"].shape[1:] == (10, 6)
    assert len(np.unique(data["episode"])) == 2

    out = tmp_path / "critic"
    summary = train(str(roll), str(out), epochs=2, ensemble=2)
    assert (out / "critic.pt").exists()
    assert summary["feature_dim"] == data["state"].shape[1] + data["pooled"].shape[1]

    device = torch.device("cuda")
    actor = build_actor("gaf", CHECKPOINT, device, DEFAULT_TASK,
                        n_exec=10, critic_dir=str(out), seed=0)
    obs = [{"front": np.zeros((480, 640, 3), np.uint8),
            "wrist": np.zeros((480, 640, 3), np.uint8),
            "state": np.zeros(6, np.float32)}]
    guided = actor.act(obs)
    assert guided.shape == (1, 10, 6)
    assert np.isfinite(guided).all()

    # Guidance must actually change the actions; if it does not, the critic
    # gradient is being masked away or the gate is stuck at zero.
    plain = build_actor("base", CHECKPOINT, device, DEFAULT_TASK, n_exec=10, seed=0)
    assert not np.allclose(guided, plain.act(obs))


@gpu_only
@needs_checkpoint
def test_evaluate_writes_a_scored_result(tmp_path):
    from grasprl.eval.evaluate import evaluate

    out = evaluate(method="base", checkpoint=CHECKPOINT, label="smoke", episodes=2,
                   seeds=(0,), split="validation", n_envs=2, max_ticks=60,
                   domain_randomize=False, results_dir=str(tmp_path), quiet=True)
    assert 0.0 <= out["success_mean"] <= 1.0
    written = json.loads((tmp_path / "eval_smoke.json").read_text())
    assert written["method"] == "base" and written["split"] == "validation"
    # Every category must be present, so a missing failure mode is visible as 0
    # rather than as an absent row in the report.
    from grasprl.envs import rules
    assert set(written["rates_mean"]) == set(rules.CATEGORIES)


def test_evaluate_rejects_mismatched_method_and_critic():
    from grasprl.policy.actor import build_actor

    device = torch.device("cpu")
    with pytest.raises(ValueError):
        build_actor("gaf", CHECKPOINT, device, "t")          # gaf without a critic
    with pytest.raises(ValueError):
        build_actor("base", CHECKPOINT, device, "t", critic_dir="x")   # critic without gaf
