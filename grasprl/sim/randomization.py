"""Domain randomization applied per episode to bridge the sim->real gap.

Mutates the compiled ``MjModel`` in place (colours, lights, camera pose/FOV,
cube friction/mass) before an episode is recorded. Proprioception/action noise
is applied by the recorder to the logged values, not the model.

Centers come from ``configs/scene.yaml``; ranges from ``configs/randomization.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from grasprl.sim.scene import Scene, _quat_lookat

_DR_CFG = Path(__file__).resolve().parents[2] / "configs" / "randomization.yaml"


def load_dr_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or _DR_CFG).read_text())


def _jitter_rgb(base, delta, rng) -> list[float]:
    rgb = np.array(base[:3]) + rng.uniform(-delta, delta, 3)
    return [*np.clip(rgb, 0.0, 1.0), base[3] if len(base) > 3 else 1.0]


class DomainRandomizer:
    """Randomises visuals + dynamics of a :class:`Scene` each episode."""

    def __init__(self, scene: Scene, cfg: dict | None = None):
        self.scene = scene
        self.cfg = cfg if cfg is not None else load_dr_config()
        self.scene_cfg = scene.cfg
        import mujoco

        self.mj = mujoco
        m = scene.model
        # Cache geom ids by category.
        self._cube_geoms = self._geoms_of_bodies(["cube_left", "cube_right"])
        self._cup_geoms = self._geoms_of_bodies(["cup"])
        self._table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._wall_geoms = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("wall_front", "wall_left", "wall_right")
        ]
        self._cam = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n)
                     for n in ("front", "wrist")}
        self._cube_nominal_mass = {
            b: float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)])
            for b in ("cube_left", "cube_right")
        }

    def _geoms_of_bodies(self, names):
        mj, m = self.mj, self.scene.model
        ids = {mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in names}
        return [gi for gi in range(m.ngeom) if m.geom_bodyid[gi] in ids]

    def apply(self, rng: np.random.Generator):
        if not self.cfg.get("enabled", True):
            return
        self._randomize_colors(rng)
        self._randomize_lighting(rng)
        self._randomize_cameras(rng)
        self._randomize_physics(rng)

    # -- individual knobs ----------------------------------------------------

    def _randomize_colors(self, rng):
        m, c = self.scene.model, self.cfg["color_jitter"]
        sc = self.scene_cfg
        for gi in self._cube_geoms:
            m.geom_rgba[gi] = _jitter_rgb(sc["cubes"]["rgba"], c["cube"], rng)
        for gi in self._cup_geoms:
            m.geom_rgba[gi] = _jitter_rgb(sc["cup"]["rgba"], c["cup"], rng)
        m.geom_rgba[self._table_geom] = _jitter_rgb(sc["table"]["rgba"], c["table"], rng)
        wall_rgba = _jitter_rgb(sc["walls"]["rgba"], c["wall"], rng)
        for gi in self._wall_geoms:
            if gi >= 0:
                m.geom_rgba[gi] = wall_rgba

    def _randomize_lighting(self, rng):
        m, L = self.scene.model, self.cfg["lighting"]
        amb = rng.uniform(*L["ambient"])
        for i in range(m.nlight):
            d = rng.uniform(*L["diffuse"])
            m.light_diffuse[i] = [d, d, d]
            m.light_ambient[i] = [amb, amb, amb]
            base = np.array(self.scene_cfg["lights"][i]["pos"]) if i < len(self.scene_cfg["lights"]) else m.light_pos[i]
            m.light_pos[i] = base + rng.uniform(-1, 1, 3) * L["pos_jitter"]

    def _randomize_cameras(self, rng):
        m = self.scene.model
        cam = self.scene_cfg["cameras"]
        fc = self.cfg["camera"]["front"]
        fpos = np.array(cam["front"]["pos"]) + rng.uniform(-1, 1, 3) * fc["pos_jitter"]
        flook = np.array(cam["front"]["lookat"]) + rng.uniform(-1, 1, 3) * fc["lookat_jitter"]
        cid = self._cam["front"]
        m.cam_pos[cid] = fpos
        m.cam_quat[cid] = _quat_lookat(fpos, flook)
        m.cam_fovy[cid] = cam["front"]["fovy"] + rng.uniform(-1, 1) * fc["fovy_jitter"]
        # Wrist camera: jitter its mounting offset + FOV (orientation kept).
        wc = self.cfg["camera"]["wrist"]
        wid = self._cam["wrist"]
        m.cam_pos[wid] = np.array(cam["wrist"]["pos"]) + rng.uniform(-1, 1, 3) * wc["pos_jitter"]
        m.cam_fovy[wid] = cam["wrist"]["fovy"] + rng.uniform(-1, 1) * wc["fovy_jitter"]

    def _randomize_physics(self, rng):
        mj, m, p = self.mj, self.scene.model, self.cfg["physics"]
        fr = rng.uniform(*p["cube_friction"])
        scale = rng.uniform(*p["cube_mass_scale"])
        for gi in self._cube_geoms:
            m.geom_friction[gi, 0] = fr
        for b, nominal in self._cube_nominal_mass.items():
            bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b)
            m.body_mass[bid] = nominal * scale
