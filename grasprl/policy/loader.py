"""Load a SmolVLA checkpoint and turn environment observations into the batches
it expects.

Two things live here:

* :func:`load_smolvla` restores the policy together with the pre- and
  post-processor pipelines saved next to it. Those pipelines carry the
  normalization statistics and the ``front -> camera1`` / ``wrist -> camera2``
  renaming that the imitation run was trained with, so loading them from the
  checkpoint (rather than rebuilding them) is what keeps RL fine-tuning in the
  exact same input and output space as imitation.
* :func:`observations_to_batch` stacks a list of environment observations into
  one batch in LeRobot's key format and runs it through the preprocessor. Batching
  matters: the SmolVLA forward pass is the bottleneck of a PPO rollout, so all
  parallel environments are encoded in a single call.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERA_KEYS = {"front": "observation.images.front", "wrist": "observation.images.wrist"}

# SmolVLA's pretrained camera slots. The imitation run passes the same map to
# lerobot-train, which bakes it into the saved processor; passing it again here
# is a no-op for those checkpoints and is what makes an un-finetuned
# lerobot/smolvla_base usable directly, which the sampler tests rely on.
DEFAULT_RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


def load_smolvla(
    checkpoint_dir: str | Path,
    device: torch.device,
    eval_mode: bool = True,
    rename_map: dict[str, str] | None = None,
):
    """Return ``(policy, preprocessor, postprocessor)`` for a SmolVLA checkpoint.

    ``checkpoint_dir`` is a LeRobot ``pretrained_model/`` directory, the format
    written by ``lerobot-train``.
    """
    from lerobot.policies.factory import make_pre_post_processors

    checkpoint_dir = str(checkpoint_dir)
    cfg_path = Path(checkpoint_dir) / "config.json"
    if cfg_path.exists():
        ptype = json.loads(cfg_path.read_text()).get("type", "").lower()
        if ptype and ptype != "smolvla":
            raise ValueError(
                f"{checkpoint_dir} holds a {ptype!r} policy; Flow-SDE PPO is "
                f"specific to SmolVLA's flow-matching sampler"
            )

    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    policy.to(device)
    policy.eval() if eval_mode else policy.train()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {
                "rename_map": DEFAULT_RENAME_MAP if rename_map is None else rename_map
            },
        },
    )
    return policy, preprocessor, postprocessor


def observations_to_batch(
    observations: list[dict],
    instruction: str,
    preprocessor,
    device: torch.device,
) -> dict:
    """Stack environment observations into one preprocessed SmolVLA batch.

    Environment images are ``(H, W, 3)`` uint8; LeRobot policies want
    ``(B, 3, H, W)`` float in [0, 1], which is what ``preprocess_observation``
    produces for gym envs. The same conversion is done here directly, since this
    env already emits LeRobot-shaped keys.
    """
    batch: dict = {}
    for cam, key in CAMERA_KEYS.items():
        imgs = np.stack([o[cam] for o in observations])
        t = torch.from_numpy(imgs).to(device)
        batch[key] = t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    states = np.stack([o["state"] for o in observations]).astype(np.float32)
    batch[OBS_STATE] = torch.from_numpy(states).to(device)
    batch["task"] = [instruction] * len(observations)
    return preprocessor(batch)


def postprocess_actions(actions: torch.Tensor, postprocessor) -> np.ndarray:
    """Unnormalize a policy action chunk back into LeRobot robot units.

    ``actions`` is ``(B, chunk, action_dim)``. The postprocessor pipeline expects
    a 2-D ``(B, action_dim)`` tensor, so the chunk axis is folded into the batch
    and restored afterwards.
    """
    b, chunk, dim = actions.shape
    out = postprocessor(actions.reshape(b * chunk, dim))
    if isinstance(out, dict):
        out = out[ACTION]
    return out.reshape(b, chunk, dim).detach().to("cpu").float().numpy()
