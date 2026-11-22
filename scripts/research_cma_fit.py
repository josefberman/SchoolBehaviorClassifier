#!/usr/bin/env python3
"""Joint CMA-ES calibration of Couzin params vs the real KS vector (10-fish)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cma
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.research_sim_realism import (
    STAT_KEYS,
    compare,
    mean_w,
    montage,
    plot_histograms,
    real_pools,
    snapshots,
)
from src.features.dataset import load_real_segments
from src.features.extended import compute_extended_series
from src.labels import BEHAVIOR_SHORT, CANONICAL, is_transition
from src.sim.config import CONFIG_DIR, DEFAULTS, deep_merge, validate_radii
from src.sim.model_fast import run_simulation_fast
from src.sim.realistic import run_realistic

OUT = ROOT / "research" / "results"
FIG = ROOT / "research" / "figures"
CMA_DIR = ROOT / "configs" / "behaviors_cma"

# Decision variables (unconstrained CMA space → positive via exp/clip).
# order: s_0, r_r, gap_o, gap_a, w_o, w_tan, w_rad, sigma_theta, omega_max, w_r, sigma_s
BOUNDS_LO = np.array([0.7, 12.0, 20.0, 30.0, 0.05, 0.0, -1.6, 0.01, 0.06, 0.4, 0.02])
BOUNDS_HI = np.array([3.4, 70.0, 90.0, 160.0, 2.4, 1.1, 1.6, 0.22, 0.55, 3.2, 0.7])


def _decode(z: np.ndarray) -> dict:
    x = BOUNDS_LO + (BOUNDS_HI - BOUNDS_LO) / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    r_r, gap_o, gap_a = float(x[1]), float(x[2]), float(x[3])
    r_o = r_r + gap_o
    r_a = r_o + gap_a
    return {
        "s_0": float(x[0]),
        "r_r": r_r,
        "r_o": r_o,
        "r_a": r_a,
        "w_o": float(x[4]),
        "w_tan": float(x[5]),
        "w_rad": float(x[6]),
        "sigma_theta": float(x[7]),
        "omega_max": float(x[8]),
        "w_r": float(x[9]),
        "sigma_s": float(x[10]),
        "w_a": 1.0,
        "speed_dep_turn": True,
        "ou_rho": 0.15,
        "fps": 29.97,
        "dt": 1.0,
        "burn_in": 280,
        "record_frames": 90,
    }


def _encode_from_yaml(cfg: dict) -> np.ndarray:
    r_r = float(cfg["r_r"])
    r_o = float(cfg["r_o"])
    r_a = float(cfg["r_a"])
    raw = np.array(
        [
            cfg["s_0"],
            r_r,
            r_o - r_r,
            r_a - r_o,
            cfg["w_o"],
            cfg["w_tan"],
            cfg["w_rad"],
            cfg["sigma_theta"],
            cfg["omega_max"],
            cfg["w_r"],
            cfg.get("sigma_s", 0.1),
        ],
        dtype=np.float64,
    )
    u = (raw - BOUNDS_LO) / np.maximum(BOUNDS_HI - BOUNDS_LO, 1e-9)
    u = np.clip(u, 1e-4, 1 - 1e-4)
    return np.log(u / (1.0 - u))


def _sim_series(lab: str, cfg: dict, seeds: int, n: int) -> dict[str, np.ndarray]:
    buckets = {k: [] for k in STAT_KEYS}
    for seed in range(seeds):
        res = run_realistic(lab, n, seed, cfg, prefer_fit=False)
        s = compute_extended_series(res.positions, res.velocities)
        for k in STAT_KEYS:
            if k in s:
                buckets[k].append(np.asarray(s[k], dtype=np.float64))
    return {k: np.concatenate(v) if v else np.array([]) for k, v in buckets.items()}


def _ks_vector(real_lab: dict, sim_lab: dict) -> dict[str, float]:
    from scipy.stats import ks_2samp

    out = {}
    rng = np.random.default_rng(0)
    for k in STAT_KEYS:
        r = np.asarray(real_lab.get(k, []), dtype=np.float64)
        s = np.asarray(sim_lab.get(k, []), dtype=np.float64)
        r, s = r[np.isfinite(r)], s[np.isfinite(s)]
        if k == "aspect":
            r, s = r[(r >= 1) & (r <= 20)], s[(s >= 1) & (s <= 20)]
        if r.size < 20 or s.size < 20:
            out[k] = 1.0
            continue
        rr = r if r.size <= 2500 else rng.choice(r, 2500, replace=False)
        ss = s if s.size <= 2500 else rng.choice(s, 2500, replace=False)
        out[k] = float(ks_2samp(rr, ss).statistic)
    return out


def _objective(lab: str, z, real_lab: dict, seeds: int, n: int) -> float:
    cfg = _decode(z)
    try:
        validate_radii(cfg)
    except ValueError:
        return 8.0
    try:
        sim = _sim_series(lab, cfg, seeds=seeds, n=n)
    except Exception:
        return 8.0
    ks = _ks_vector(real_lab, sim)
    # Core vector: speed, scale, order. Aspect/entropy secondary.
    core = ("mean_speed", "nnd", "R_gyr", "phi_trans", "M", "phi_rad_pm")
    loss = float(np.mean([ks[k] for k in core]))
    nnd = float(np.nanmean(sim["nnd"])) if sim["nnd"].size else 1e9
    R = float(np.nanmean(sim["R_gyr"])) if sim["R_gyr"].size else 1e9
    if nnd > 220 or R > 480 or nnd < 8:
        loss += 2.5
    if lab == "milling":
        M = float(np.nanmean(sim["M"])) if sim["M"].size else 0.0
        loss += 1.8 * max(0.0, 0.50 - M)
    if lab == "expansion":
        pr = float(np.nanmean(sim["phi_rad_pm"])) if sim["phi_rad_pm"].size else 0.0
        loss += 1.2 * max(0.0, 0.15 - pr)
    if lab == "compaction":
        pr = float(np.nanmean(sim["phi_rad_pm"])) if sim["phi_rad_pm"].size else 0.0
        loss += 1.2 * max(0.0, pr + 0.12)
    if lab == "traveling":
        pt = float(np.nanmean(sim["phi_trans"])) if sim["phi_trans"].size else 0.0
        loss += 1.0 * max(0.0, 0.70 - pt)
    return float(loss)


def fit_label(lab: str, real_lab: dict, seeds: int, n: int, maxiter: int, popsize: int) -> dict:
    with open(CONFIG_DIR / f"{BEHAVIOR_SHORT[lab]}.yaml", encoding="utf-8") as f:
        orig = yaml.safe_load(f) or {}
    x0 = _encode_from_yaml(orig)
    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        0.45,
        {"popsize": popsize, "maxiter": maxiter, "verbose": -9, "seed": 7},
    )
    history = []
    while not es.stop():
        xs = es.ask()
        vals = [_objective(lab, x, real_lab, seeds, n) for x in xs]
        es.tell(xs, vals)
        history.append({"best": float(es.result.fbest), "mean": float(np.mean(vals))})
        print(f"  {lab} iter {len(history):02d}  best={es.result.fbest:.3f}  mean={np.mean(vals):.3f}", flush=True)
    zbest = np.asarray(es.result.xbest, dtype=np.float64)
    cfg = _decode(zbest)
    cfg["behavior"] = lab
    return {
        "loss": float(es.result.fbest),
        "history": history,
        "cfg": cfg,
        "n_evals": int(es.result.evaluations),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--maxiter", type=int, default=10)
    p.add_argument("--popsize", type=int, default=8)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CMA_DIR.mkdir(parents=True, exist_ok=True)

    print("pooling real…", flush=True)
    real = real_pools()

    fits = {}
    for lab in CANONICAL:
        print(f"CMA-ES {lab}", flush=True)
        if lab not in real:
            continue
        fits[lab] = fit_label(lab, real[lab], args.seeds, args.n, args.maxiter, args.popsize)
        short = BEHAVIOR_SHORT[lab]
        with open(CMA_DIR / f"{short}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(fits[lab]["cfg"], f, sort_keys=False)

    print("baseline sims…", flush=True)
    from collections import defaultdict

    def _run_cma(lab: str, n: int, seed: int):
        short = BEHAVIOR_SHORT[lab]
        with open(CMA_DIR / f"{short}.yaml", encoding="utf-8") as f:
            ov = yaml.safe_load(f) or {}
        return run_realistic(lab, n, seed, overrides=ov, prefer_fit=False)

    def _pool_engine(engine: str):
        buckets = defaultdict(lambda: defaultdict(list))
        for lab in CANONICAL:
            for seed in range(6):
                if engine == "baseline":
                    res = run_simulation_fast(lab, args.n, seed)
                else:
                    res = _run_cma(lab, args.n, seed)
                s = compute_extended_series(res.positions, res.velocities)
                for k in STAT_KEYS:
                    if k in s:
                        buckets[lab][k].append(np.asarray(s[k], dtype=np.float64))
        return {lab: {k: np.concatenate(v) for k, v in d.items()} for lab, d in buckets.items()}

    before = _pool_engine("baseline")
    after = _pool_engine("cma")
    before_cmp = compare(real, before)
    after_cmp = compare(real, after)

    segs = load_real_segments()
    by_lab = defaultdict(list)
    for s in segs:
        if not is_transition(s["label"]):
            by_lab[s["label"]].append(s)
    snapshots(
        by_lab,
        n=args.n,
        fit_prefix="sim_cma",
        fit_title="sim CMA",
        run_fit=_run_cma,
    )
    plot_histograms(real, before, after, FIG / "sim_cma_histograms.png")
    montage(
        FIG / "sim_cma_montage.png",
        cols=("real", "sim_old", "sim_cma"),
        titles={
            "real": "real",
            "sim_old": "sim (original YAML)",
            "sim_cma": "sim (CMA-ES)",
        },
    )

    report = {
        "method": "CMA-ES on logit-constrained (s_0,r_r,r_o,r_a,w_o,w_tan,w_rad,sigma_theta,omega_max,w_r,sigma_s)",
        "seeds_eval": 6,
        "cma_seeds_per_eval": args.seeds,
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "fits": {k: {"loss": v["loss"], "n_evals": v["n_evals"], "cfg": v["cfg"], "history": v["history"]} for k, v in fits.items()},
        "baseline_mean_ks": mean_w(before_cmp, "ks_stat"),
        "cma_mean_ks": mean_w(after_cmp, "ks_stat"),
        "baseline_mean_w_over_std": mean_w(before_cmp, "wasserstein_over_std"),
        "cma_mean_w_over_std": mean_w(after_cmp, "wasserstein_over_std"),
        "baseline": before_cmp,
        "cma": after_cmp,
    }
    (OUT / "cma_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"mean KS  baseline={report['baseline_mean_ks']:.4f}  CMA={report['cma_mean_ks']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
