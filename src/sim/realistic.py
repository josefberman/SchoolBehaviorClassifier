"""Optional realistic simulator extras. Original FastSchoolSimulator is unchanged.

Adds (all opt-in via cfg keys that default YAMLs do not set):

* ``speed_dep_turn`` — ω_max scales as s_0 / (speed + eps)
* ``ou_rho`` — AR(1) heading and speed noise instead of iid
* ``init_span`` — initial blob size as a fraction of arena half-extent
* ``init_from`` — burn in under another behavior before switching parameters
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.sim.config import BEHAVIOR_KEYS, deep_merge, load_behavior_config, validate_radii
from src.sim.model import SimResult, _wrap
from src.sim.model_fast import FastSchoolSimulator


class RealisticSchoolSimulator(FastSchoolSimulator):
    def __init__(self, n: int, cfg: dict[str, Any], seed: int = 0):
        super().__init__(n, cfg, seed)
        self.ou_rho = float(cfg.get("ou_rho", 0.0))
        self.speed_dep_turn = bool(cfg.get("speed_dep_turn", False))
        self._eps_w = np.zeros(self.n)
        self._eps_a = np.zeros(self.n)
        span_frac = float(cfg.get("init_span", 0.35))
        if span_frac != 0.35:
            span = span_frac * self.half_extent
            self.pos[:, 0] = self.center[0] + self.rng.uniform(-span, span, self.n)
            self.pos[:, 1] = self.center[1] + self.rng.uniform(-span, span, self.n)

    def step(self) -> None:
        cfg = self.cfg
        t_r, t_o, t_a = self._social()
        t_tan, t_rad = self._centroid_steering()
        z_w = self.rng.normal(0.0, 1.0, self.n)
        z_a = self.rng.normal(0.0, 1.0, self.n)
        rho = self.ou_rho
        if rho > 0.0:
            s = np.sqrt(max(1.0 - rho**2, 0.0))
            self._eps_w = rho * self._eps_w + s * z_w
            self._eps_a = rho * self._eps_a + s * z_a
            eps_w, eps_a = self._eps_w, self._eps_a
        else:
            eps_w, eps_a = z_w, z_a

        omega_max = float(cfg["omega_max"])
        if self.speed_dep_turn:
            s0 = max(float(cfg["s_0"]), 1e-6)
            omega_max = omega_max * (s0 / np.maximum(self.speed, 0.15 * s0))

        omega = (
            float(cfg["w_r"]) * t_r
            + float(cfg["w_o"]) * t_o
            + float(cfg["w_a"]) * t_a
            + float(cfg["w_tan"]) * t_tan
            + float(cfg["w_rad"]) * t_rad
            + float(cfg["sigma_theta"]) * eps_w
            + self._wall()
        )
        omega = np.clip(omega, -omega_max, omega_max)
        self.theta = np.mod(self.theta + omega, 2.0 * np.pi)

        delta_s = np.clip(
            float(cfg["s_0"]) - self.speed + float(cfg["sigma_s"]) * eps_a,
            -float(cfg["a_max"]),
            float(cfg["a_max"]),
        )
        self.speed = np.maximum(0.0, self.speed + delta_s)
        self.pos = self.pos + self.velocities() * self.dt
        self._clip_to_arena()
        self.frame += 1


def _load(behavior: str, overrides: dict | None = None) -> dict:
    from src.labels import BEHAVIOR_SHORT

    short = BEHAVIOR_SHORT.get(behavior, behavior)
    # Prefer fitted configs when present.
    from pathlib import Path

    from src.sim.config import CONFIG_DIR, DEFAULTS, deep_merge as dm

    fit = Path(CONFIG_DIR).parent / "behaviors_fit" / f"{short}.yaml"
    src = fit if fit.exists() else (CONFIG_DIR / f"{short}.yaml")
    import yaml

    with open(src, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = dm(DEFAULTS, data)
    if overrides:
        cfg = dm(cfg, overrides)
    validate_radii(cfg)
    cfg["behavior"] = behavior
    return cfg


def run_realistic(
    behavior: str,
    n: int,
    seed: int,
    overrides: dict | None = None,
    *,
    init_from: str | None = None,
) -> SimResult:
    cfg = _load(behavior, overrides)
    init_from = init_from or cfg.get("init_from")
    burn = int(cfg["burn_in"])
    T = int(cfg["record_frames"])
    if init_from:
        cfg_init = _load(str(init_from), None)
        for k in ("half_extent_x", "half_extent_y"):
            cfg_init["arena"][k] = cfg["arena"][k]
        cfg_init["burn_in"] = burn
        cfg_init["record_frames"] = T
        for key in ("ou_rho", "speed_dep_turn", "fps", "dt"):
            if key in cfg:
                cfg_init[key] = cfg[key]
        sim = RealisticSchoolSimulator(n=n, cfg=cfg_init, seed=seed)
        for _ in range(burn):
            sim.step()
        for k in BEHAVIOR_KEYS:
            sim.cfg[k] = cfg[k]
        sim.cfg["behavior"] = behavior
    else:
        sim = RealisticSchoolSimulator(n=n, cfg=cfg, seed=seed)
        for _ in range(burn):
            sim.step()

    pos_hist = np.zeros((T, n, 2))
    vel_hist = np.zeros((T, n, 2))
    sim.frame = 0
    for t in range(T):
        sim.step()
        pos_hist[t] = sim.pos
        vel_hist[t] = sim.velocities()
    return SimResult(
        positions=pos_hist,
        velocities=vel_hist,
        meta={"n": n, "behavior": behavior, "fps": sim.fps, "engine": "realistic"},
    )
