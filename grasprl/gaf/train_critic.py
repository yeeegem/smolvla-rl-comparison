"""Train the Guided Action Flow critic ensemble (paper Sec. III-D/E, Eq. 4-5).

Target -- sparse success-to-go, Eq. (4)::

    y_i = gamma^(j* - i)   where j* = min{ j >= i : s_j = 1 }
    y_i = 0                if the episode never succeeds

so a chunk is worth ``gamma^k`` if success arrives k decisions later, and nothing
at all in a failed episode. It is cheap to compute and needs no bootstrapping,
but the paper is candid that it is also the method's weakest link: it does not
finely rank chunks that are all *nearly* right, which is precisely the regime a
grasp-stability critic lives in.

Loss -- plain regression, Eq. (5): ``E[(Q_phi(f_o, a, e_tau) - y)^2]``.

**The split is by episode, never by chunk.** Chunks from one rollout overlap and
share trajectory context, so a random chunk split leaks the validation episodes
into training and reports a critic that looks far better than it is. The paper
calls this out; :func:`split_by_episode` enforces it and
``tests/test_gaf.py::test_split_is_episode_level`` pins it.

Usage::

    uv run python -m grasprl.gaf.train_critic \\
        --rollouts recordings/gaf_rollouts --out runs/gaf_critic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from grasprl.gaf.critic import CriticConfig, CriticEnsemble, build_features, feature_dim

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def success_to_go(success: np.ndarray, episode: np.ndarray, gamma: float) -> np.ndarray:
    """Eq. (4), vectorised per episode.

    ``success[i]`` marks the decision at which the episode was first scored a
    success. Everything at or after it is worth 1; earlier steps decay backwards
    by ``gamma``; a failed episode is all zeros.
    """
    y = np.zeros(len(success), dtype=np.float32)
    for ep in np.unique(episode):
        idx = np.flatnonzero(episode == ep)
        hits = np.flatnonzero(success[idx] == 1)
        if len(hits) == 0:
            continue
        j = hits[0]
        k = np.arange(len(idx))
        y[idx] = np.where(k <= j, gamma ** (j - k), 1.0)
    return y


def split_by_episode(episode: np.ndarray, val_fraction: float = 0.2, seed: int = 0):
    """Disjoint train/val index arrays whose episodes never overlap."""
    eps = np.unique(episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_val = max(1, round(len(eps) * val_fraction))
    val_eps = set(eps[:n_val].tolist())
    is_val = np.array([e in val_eps for e in episode])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def load_critic(path: str | Path, device: torch.device) -> CriticEnsemble:
    """Restore a trained ensemble, config included."""
    path = Path(path)
    blob = torch.load(path / "critic.pt", map_location=device, weights_only=False)
    cfg = CriticConfig(**blob["config"])
    ens = CriticEnsemble(blob["feature_dim"], cfg)
    ens.load_state_dict(blob["state_dict"])
    return ens.to(device).eval()


def train(rollouts: str, out_dir: str, config_path: str | None = None,
          seed: int = 0, **overrides) -> dict:
    raw = yaml.safe_load(Path(config_path or _CONFIGS / "gaf.yaml").read_text())
    cfg = CriticConfig(**{**raw.get("critic", {}), **overrides})

    data = np.load(Path(rollouts) / "rollouts.npz")
    state, pooled = data["state"], data["pooled"]
    actions, episode = data["action"], data["episode"]
    y = success_to_go(data["success"], episode, cfg.gamma)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = build_features(
        cfg,
        torch.as_tensor(state), torch.as_tensor(pooled), None,
    ).to(device)
    acts = torch.as_tensor(actions[:, : cfg.horizon, : cfg.action_dim]).to(device)
    targets = torch.as_tensor(y).to(device)

    tr_idx, va_idx = split_by_episode(episode, seed=seed)
    fdim = feature_dim(cfg, state.shape[1], pooled.shape[1])
    assert feats.shape[1] == fdim, (feats.shape, fdim)
    print(f"samples {len(y)} ({len(tr_idx)} train / {len(va_idx)} val over "
          f"{len(np.unique(episode))} episodes) | feature_dim {fdim} "
          f"| positives {float((targets > 0).float().mean()):.3f}", flush=True)

    ens = CriticEnsemble(fdim, cfg, seed=seed).to(device)
    opt = torch.optim.AdamW(ens.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    rng = np.random.default_rng(seed)

    for epoch in range(cfg.epochs):
        ens.train()
        order = rng.permutation(tr_idx)
        total, nb = 0.0, 0
        for s in range(0, len(order), cfg.batch_size):
            idx = torch.as_tensor(order[s:s + cfg.batch_size], device=device)
            # Every member sees the same batch; they differ only by
            # initialisation. That is enough to make their disagreement a usable
            # signal, and it keeps training a single cheap pass.
            q = ens(feats[idx], acts[idx])
            loss = ((q - targets[idx].unsqueeze(0)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        ens.eval()
        with torch.no_grad():
            vi = torch.as_tensor(va_idx, device=device)
            qm, qs = ens.mean_and_std(feats[vi], acts[vi])
            val_mse = float(((qm - targets[vi]) ** 2).mean())
            disagree = float(qs.mean())
        history.append({"epoch": epoch + 1, "train_mse": total / max(nb, 1),
                        "val_mse": val_mse, "val_ensemble_std": disagree})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3}/{cfg.epochs} train {total / max(nb,1):.5f} "
                  f"val {val_mse:.5f} disagreement {disagree:.4f}", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ens.state_dict(), "config": asdict(cfg),
                "feature_dim": fdim}, out / "critic.pt")
    summary = {
        "rollouts": str(rollouts), "samples": len(y),
        "episodes": len(np.unique(episode)),
        "train_samples": len(tr_idx), "val_samples": len(va_idx),
        "feature_dim": fdim, "config": asdict(cfg),
        "final_train_mse": history[-1]["train_mse"],
        "final_val_mse": history[-1]["val_mse"],
        "final_ensemble_std": history[-1]["val_ensemble_std"],
    }
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", default="recordings/gaf_rollouts")
    p.add_argument("--out", default="runs/gaf_critic")
    p.add_argument("--config", default=None, help="default configs/gaf.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--critic-features", default=None, choices=["state", "state+pooled"],
                   help="paper-faithful compact critic vs the pooled-VLM variant")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--ensemble", type=int, default=None)
    a = p.parse_args(argv)
    ov = {}
    if a.critic_features:
        ov["features"] = a.critic_features
    if a.epochs:
        ov["epochs"] = a.epochs
    if a.ensemble:
        ov["ensemble"] = a.ensemble
    train(rollouts=a.rollouts, out_dir=a.out, config_path=a.config, seed=a.seed, **ov)


if __name__ == "__main__":
    main()
