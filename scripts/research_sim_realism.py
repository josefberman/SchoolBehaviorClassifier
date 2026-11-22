#!/usr/bin/env python3
"""Diagnose sim vs real kinematics and retune YAML from annotated_config medians."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.dataset import load_positions_velocities, load_real_segments
from src.features.extended import compute_extended_series
from src.labels import BEHAVIOR_SHORT, CANONICAL, is_transition
from src.sim.config import CONFIG_DIR, DEFAULTS, deep_merge
from src.sim.model_fast import run_simulation_fast
from src.sim.realistic import run_realistic
from src.sim.render import save_still

OUT = ROOT / "research" / "results"
FIG = ROOT / "research" / "figures"
FIT_DIR = ROOT / "configs" / "behaviors_fit"

STAT_KEYS = (
    "mean_speed",
    "nnd",
    "phi_trans",
    "M",
    "phi_rad_pm",
    "aspect",
    "R_gyr",
    "heading_entropy",
    "dlogR",
)


def _series_pool(pos, vel) -> dict[str, np.ndarray]:
    s = compute_extended_series(pos, vel)
    return {k: np.asarray(s[k], dtype=np.float64) for k in STAT_KEYS if k in s}


def real_pools() -> dict[str, dict[str, np.ndarray]]:
    segs = load_real_segments()
    loaded = {}
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in segs:
        if is_transition(s["label"]):
            continue
        key = s["csv"]
        if key not in loaded:
            loaded[key] = load_positions_velocities(ROOT / key)
        pos, vel = loaded[key]
        a, b = s["start"], min(s["end"], pos.shape[0])
        if b - a < 10:
            continue
        pool = _series_pool(pos[a:b], vel[a:b])
        for k, arr in pool.items():
            buckets[s["label"]][k].append(arr)
    out = {}
    for lab, d in buckets.items():
        out[lab] = {k: np.concatenate(v) for k, v in d.items()}
    return out


def sim_pools(engine: str, n_list: list[int], seeds: int, overrides_by: dict | None = None):
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for lab in CANONICAL:
        ov = (overrides_by or {}).get(lab, {})
        for n in n_list:
            for seed in range(seeds):
                if engine == "baseline":
                    res = run_simulation_fast(lab, n, seed)
                else:
                    res = run_realistic(lab, n, seed, ov)
                pool = _series_pool(res.positions, res.velocities)
                for k, arr in pool.items():
                    buckets[lab][k].append(arr)
    return {lab: {k: np.concatenate(v) for k, v in d.items()} for lab, d in buckets.items()}


def _finite(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def compare(real, sim) -> dict:
    report = {}
    for lab in sorted(set(real) | set(sim)):
        report[lab] = {}
        for k in STAT_KEYS:
            r = _finite(real.get(lab, {}).get(k, []))
            s = _finite(sim.get(lab, {}).get(k, []))
            if r.size == 0 or s.size == 0:
                report[lab][k] = {"n_real": int(r.size), "n_sim": int(s.size)}
                continue
            # subsample for KS speed
            rng = np.random.default_rng(0)
            rr = r if r.size <= 8000 else rng.choice(r, 8000, replace=False)
            ss = s if s.size <= 8000 else rng.choice(s, 8000, replace=False)
            ks = ks_2samp(rr, ss)
            report[lab][k] = {
                "n_real": int(r.size),
                "n_sim": int(s.size),
                "real_mean": float(np.mean(r)),
                "real_std": float(np.std(r)),
                "sim_mean": float(np.mean(s)),
                "sim_std": float(np.std(s)),
                "wasserstein": float(wasserstein_distance(rr, ss)),
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
            }
    return report


def mean_w(report: dict) -> float:
    vals = []
    for lab in report:
        for k in report[lab]:
            if "wasserstein" in report[lab][k]:
                vals.append(report[lab][k]["wasserstein"])
    return float(np.mean(vals)) if vals else float("nan")


def write_fit_yamls() -> dict[str, dict]:
    """Median inverse-fitted parameters per stable label → configs/behaviors_fit/."""
    df = pd.read_csv(ROOT / "annotated_config.csv")
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    medians = {}
    param_cols = [
        "w_r",
        "w_o",
        "w_a",
        "w_tan",
        "w_rad",
        "sigma_theta",
        "s_0",
        "sigma_s",
        "omega_max",
        "a_max",
        "r_r",
        "r_o",
        "r_a",
    ]
    for lab in CANONICAL:
        sub = df[df["label"] == lab]
        if sub.empty:
            continue
        med = {c: float(np.nanmedian(sub[c])) for c in param_cols if c in sub.columns}
        # Keep exclusive Couzin radii; fall back to locked 50/100/150.
        r_r, r_o, r_a = med.get("r_r", 50.0), med.get("r_o", 100.0), med.get("r_a", 150.0)
        if not (0 < r_r < r_o < r_a):
            r_r, r_o, r_a = 50.0, 100.0, 150.0
        short = BEHAVIOR_SHORT[lab]
        cfg = {
            "behavior": lab if lab != "traveling" else "traveling",
            "r_r": 50.0,
            "r_o": 100.0,
            "r_a": 150.0,
            "w_r": max(med.get("w_r", 1.0), 0.0),
            "w_o": max(med.get("w_o", 1.0), 0.0),
            "w_a": max(med.get("w_a", 0.8), 0.0),
            "w_tan": max(med.get("w_tan", 0.0), 0.0),
            "w_rad": med.get("w_rad", 0.0),
            "sigma_theta": float(np.clip(med.get("sigma_theta", 0.08), 0.01, 0.4)),
            "s_0": float(np.clip(med.get("s_0", 1.5), 0.3, 4.0)),
            "sigma_s": float(np.clip(med.get("sigma_s", 0.1), 0.0, 1.5)),
            "omega_max": float(np.clip(med.get("omega_max", 0.35), 0.05, 1.2)),
            "a_max": float(np.clip(med.get("a_max", 0.8), 0.05, 2.5)),
            "fps": 29.97,
            "dt": 1.0,
            "burn_in": 400 if lab in ("expansion", "compaction") else 600,
            "record_frames": 180,
            "speed_dep_turn": True,
            "ou_rho": 0.6,
            "init_span": 0.22,
        }
        if lab in ("expansion", "compaction"):
            cfg["init_from"] = "traveling"
            cfg["burn_in"] = 500
        if lab == "milling":
            cfg["arena"] = {"wall_margin": 55.0, "w_wall": 3.5}
        medians[lab] = {"from_data": med, "written": cfg}
        with open(FIT_DIR / f"{short}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    return medians


def plot_histograms(real, before, after, path: Path) -> None:
    labs = [l for l in CANONICAL if l in real]
    keys = ["mean_speed", "nnd", "phi_trans", "M", "aspect", "phi_rad_pm"]
    fig, axes = plt.subplots(len(keys), len(labs), figsize=(3.2 * len(labs), 2.4 * len(keys)))
    if len(labs) == 1:
        axes = np.array(axes).reshape(len(keys), 1)
    for i, k in enumerate(keys):
        for j, lab in enumerate(labs):
            ax = axes[i, j]
            for arr, name, col in (
                (real.get(lab, {}).get(k, []), "real", "#222222"),
                (before.get(lab, {}).get(k, []), "sim old", "#d35400"),
                (after.get(lab, {}).get(k, []), "sim fit", "#1f77b4"),
            ):
                x = _finite(arr)
                if x.size == 0:
                    continue
                ax.hist(x, bins=40, density=True, histtype="step", label=name, color=col, lw=1.4)
            if i == 0:
                ax.set_title(lab, fontsize=10)
            if j == 0:
                ax.set_ylabel(k, fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def snapshots(real_segs_by_lab, n: int = 10) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for lab in CANONICAL:
        # real snapshot: longest available segment
        segs = real_segs_by_lab.get(lab, [])
        if segs:
            s = max(segs, key=lambda z: z["end"] - z["start"])
            key = s["csv"]
            if key not in loaded:
                loaded[key] = load_positions_velocities(ROOT / key)
            pos, vel = loaded[key]
            a, b = s["start"], min(s["end"], pos.shape[0])
            mid = a + (b - a) // 2
            save_still(
                pos[a:b],
                vel[a:b],
                FIG / f"real_{lab}.png",
                frame=mid - a,
                title=f"real {lab}",
            )
        try:
            old = run_simulation_fast(lab, n, seed=0)
            save_still(old.positions, old.velocities, FIG / f"sim_old_{lab}.png", title=f"sim old {lab}")
            new = run_realistic(lab, n, seed=0)
            save_still(new.positions, new.velocities, FIG / f"sim_fit_{lab}.png", title=f"sim fit {lab}")
        except Exception as e:
            print(f"snapshot {lab} failed: {e}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--n", type=int, nargs="*", default=[10])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("pooling real annotated frames…", flush=True)
    real = real_pools()
    print("real labels", {k: {kk: int(np.isfinite(vv).sum()) for kk, vv in d.items()} for k, d in real.items()}, flush=True)

    print("baseline sims…", flush=True)
    before = sim_pools("baseline", args.n, args.seeds)
    before_cmp = compare(real, before)

    print("writing fitted YAMLs…", flush=True)
    medians = write_fit_yamls()

    print("fitted sims…", flush=True)
    after = sim_pools("realistic", args.n, args.seeds)
    after_cmp = compare(real, after)

    segs = load_real_segments()
    by_lab = defaultdict(list)
    for s in segs:
        if not is_transition(s["label"]):
            by_lab[s["label"]].append(s)
    snapshots(by_lab, n=args.n[0] if args.n else 10)
    plot_histograms(real, before, after, FIG / "sim_real_histograms.png")

    report = {
        "n_values": args.n,
        "seeds": args.seeds,
        "baseline_mean_wasserstein": mean_w(before_cmp),
        "fitted_mean_wasserstein": mean_w(after_cmp),
        "baseline": before_cmp,
        "fitted": after_cmp,
        "fit_medians": medians,
    }
    (OUT / "sim_realism.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"mean Wasserstein  baseline={report['baseline_mean_wasserstein']:.4f}  "
        f"fitted={report['fitted_mean_wasserstein']:.4f}"
    )
    print(f"wrote {OUT / 'sim_realism.json'}")


if __name__ == "__main__":
    main()
