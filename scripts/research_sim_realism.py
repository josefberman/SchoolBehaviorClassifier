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
from src.sim.render import save_still, save_video

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
            if k == "aspect":
                r = r[(r >= 1.0) & (r <= 20.0)]
                s = s[(s >= 1.0) & (s <= 20.0)]
            if r.size == 0 or s.size == 0:
                report[lab][k] = {"n_real": int(r.size), "n_sim": int(s.size)}
                continue
            rng = np.random.default_rng(0)
            rr = r if r.size <= 8000 else rng.choice(r, 8000, replace=False)
            ss = s if s.size <= 8000 else rng.choice(s, 8000, replace=False)
            ks = ks_2samp(rr, ss)
            rstd = float(np.std(r))
            w = float(wasserstein_distance(rr, ss))
            report[lab][k] = {
                "n_real": int(r.size),
                "n_sim": int(s.size),
                "real_mean": float(np.mean(r)),
                "real_std": rstd,
                "sim_mean": float(np.mean(s)),
                "sim_std": float(np.std(s)),
                "wasserstein": w,
                "wasserstein_over_std": float(w / max(rstd, 1e-9)),
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
            }
    return report


def mean_w(report: dict, key: str = "wasserstein") -> float:
    vals = []
    for lab in report:
        for k in report[lab]:
            if key in report[lab][k]:
                vals.append(report[lab][k][key])
    return float(np.mean(vals)) if vals else float("nan")


def write_fit_yamls() -> dict[str, dict]:
    """Speed-matched configs: original Couzin weights *and* radii, real s_0.

    Two failed alternatives (documented in REPORT.md):
    1. Dropping window-level Tobit MLE weights into YAML (w_o ~ 0.08) destroys
       cohesion — schools explode to the tank wall.
    2. Scaling radii to real NND (~50 px) while keeping original weights
       matches NND but kills milling (M 0.73 → 0.37) and expansion Φ_rad.

    Speed and interaction scale are not independently tunable. This fit only
    matches preferred speed / fps / OU heading noise, which is the change that
    improves KS without collapsing the order-parameter signatures.
    """
    df = pd.read_csv(ROOT / "annotated_config.csv")
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    medians = {}
    kin_cols = ("s_0", "sigma_s", "omega_max", "a_max")
    for lab in CANONICAL:
        short = BEHAVIOR_SHORT[lab]
        with open(CONFIG_DIR / f"{short}.yaml", encoding="utf-8") as f:
            orig = yaml.safe_load(f) or {}
        sub = df[df["label"] == lab]
        med = {c: float(np.nanmedian(sub[c])) for c in kin_cols if c in sub.columns} if len(sub) else {}
        cfg = dict(orig)
        cfg["behavior"] = lab
        if med:
            cfg["s_0"] = float(np.clip(med["s_0"], 0.6, 3.5))
            cfg["sigma_s"] = float(np.clip(med.get("sigma_s", cfg.get("sigma_s", 0.1)), 0.02, 0.8))
            # keep original omega_max / a_max / radii / social weights
        cfg["fps"] = 29.97
        cfg["dt"] = 1.0
        cfg["speed_dep_turn"] = True
        cfg["ou_rho"] = 0.2
        cfg["record_frames"] = int(orig.get("record_frames", 150))
        cfg["burn_in"] = int(orig.get("burn_in", 600))
        medians[lab] = {"from_data_kinematics": med, "written": cfg}
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
            if lab in ("traveling", "milling", "expansion"):
                save_video(
                    old.positions,
                    old.velocities,
                    FIG / f"sim_old_{lab}.gif",
                    fps=12.0,
                    stride=3,
                    title=f"sim old {lab}",
                    dpi=72,
                    figsize=(5.0, 3.0),
                )
                save_video(
                    new.positions,
                    new.velocities,
                    FIG / f"sim_fit_{lab}.gif",
                    fps=12.0,
                    stride=3,
                    title=f"sim fit {lab}",
                    dpi=72,
                    figsize=(5.0, 3.0),
                )
                if segs:
                    save_video(
                        pos[a:b],
                        vel[a:b],
                        FIG / f"real_{lab}.gif",
                        fps=12.0,
                        stride=3,
                        title=f"real {lab}",
                        dpi=72,
                        figsize=(5.0, 3.0),
                    )
        except Exception as e:
            print(f"snapshot {lab} failed: {e}", flush=True)


def montage(path: Path) -> None:
    labs = list(CANONICAL)
    cols = ("real", "sim_old", "sim_fit")
    fig, axes = plt.subplots(len(labs), 3, figsize=(9.6, 2.15 * len(labs)))
    for i, lab in enumerate(labs):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            fp = FIG / f"{col}_{lab}.png"
            ax.axis("off")
            if fp.exists():
                ax.imshow(plt.imread(fp))
            if i == 0:
                ax.set_title({"real": "real", "sim_old": "sim (original YAML)", "sim_fit": "sim (kinematic fit)"}[col], fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


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
    montage(FIG / "sim_real_montage.png")

    report = {
        "n_values": args.n,
        "seeds": args.seeds,
        "fit_method": "original Couzin weights+radii; s_0/sigma_s from real medians; speed-dep turn + OU noise",
        "baseline_mean_wasserstein": mean_w(before_cmp),
        "fitted_mean_wasserstein": mean_w(after_cmp),
        "baseline_mean_ks": mean_w(before_cmp, "ks_stat"),
        "fitted_mean_ks": mean_w(after_cmp, "ks_stat"),
        "baseline_mean_w_over_std": mean_w(before_cmp, "wasserstein_over_std"),
        "fitted_mean_w_over_std": mean_w(after_cmp, "wasserstein_over_std"),
        "baseline": before_cmp,
        "fitted": after_cmp,
        "fit_medians": medians,
    }
    (OUT / "sim_realism.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"mean KS        baseline={report['baseline_mean_ks']:.4f}  fitted={report['fitted_mean_ks']:.4f}\n"
        f"mean W/σ_real  baseline={report['baseline_mean_w_over_std']:.4f}  fitted={report['fitted_mean_w_over_std']:.4f}\n"
        f"mean W         baseline={report['baseline_mean_wasserstein']:.4f}  fitted={report['fitted_mean_wasserstein']:.4f}"
    )
    print(f"wrote {OUT / 'sim_realism.json'}")


if __name__ == "__main__":
    main()
