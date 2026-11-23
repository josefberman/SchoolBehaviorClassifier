#!/usr/bin/env python3
"""Leakage-safe feature ablations on real CouzinLab windows (grouped by video)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.grouped import binary_transient, collapse_transients, grouped_cv
from src.features.dataset import load_positions_velocities, load_real_segments
from src.features.extended import (
    all_window_feature_names,
    columns_for_families,
    extract_windows,
    family_feature_names,
)
from src.labels import CANONICAL, is_transition

OUT = ROOT / "research" / "results"
CACHE = ROOT / "research" / "cache"

FAMILIES = (
    "baseline",
    "fluct",
    "rate",
    "shape",
    "tunstrom",
    "spectral",
    "corr",
    "info",
    "fields",
    "early",
)

ABLATIONS = {
    "baseline": ("baseline",),
    "baseline+fluct": ("baseline", "fluct"),
    "baseline+rate": ("baseline", "rate"),
    "baseline+shape": ("baseline", "shape"),
    "baseline+tunstrom": ("baseline", "tunstrom"),
    "baseline+spectral": ("baseline", "spectral"),
    "baseline+corr": ("baseline", "corr"),
    "baseline+info": ("baseline", "info"),
    "baseline+fields": ("baseline", "fields"),
    "baseline+early": ("baseline", "early"),
    "physical": ("baseline", "fluct", "rate", "shape", "tunstrom", "fields", "early"),
    "nonphysical": ("spectral", "corr", "info"),
    "all": FAMILIES,
}


def _cache_path(window: int, hop: int) -> Path:
    return CACHE / f"real_windows_w{window}_h{hop}.npz"


def build_or_load(window: int, hop: int, rebuild: bool = False):
    path = _cache_path(window, hop)
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return (
            z["X"],
            z["y"],
            z["groups"],
            [str(n) for n in z["names"]],
            z["meta"],
        )
    segs = load_real_segments()
    if not segs:
        raise RuntimeError("No real segments with trajectories on disk")
    names = all_window_feature_names()
    xs, ys, gs, meta = [], [], [], []
    loaded = {}
    for s in segs:
        key = s["csv"]
        if key not in loaded:
            loaded[key] = load_positions_velocities(ROOT / key)
        pos, vel = loaded[key]
        a, b = s["start"], min(s["end"], pos.shape[0])
        if b - a < window:
            continue
        Xw, names, bounds = extract_windows(
            pos[a:b], vel[a:b], window_frames=window, hop_frames=hop, names=names
        )
        for row, (w0, w1) in zip(Xw, bounds):
            xs.append(row)
            ys.append(s["label"])
            gs.append(s["dataset"])
            meta.append((s["dataset"], s["label"], a + w0, a + w1))
    X = np.vstack(xs) if xs else np.zeros((0, len(names)))
    y = np.array(ys, dtype=object)
    groups = np.array(gs, dtype=object)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        groups=groups,
        names=np.array(names, dtype=object),
        meta=np.array(meta, dtype=object),
    )
    return X, y, groups, names, np.array(meta, dtype=object)


def _select(X, names, families):
    cols = columns_for_families(families, names)
    return X[:, cols], [names[i] for i in cols]


def _run_task(X, y, groups, names, families, model, permute_repeats=0):
    Xs, fn = _select(X, names, families)
    return grouped_cv(
        Xs, y, groups, model=model, feature_names=fn, permute_repeats=permute_repeats
    ), fn


def run(window: int, hop: int, rebuild: bool = False) -> dict:
    X, y, groups, names, _meta = build_or_load(window, hop, rebuild=rebuild)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "window": window,
        "hop": hop,
        "n_windows": int(len(y)),
        "n_videos": int(len(set(groups.tolist()))),
        "videos": sorted(set(groups.tolist()), key=str),
        "label_counts": {lab: int(np.sum(y == lab)) for lab in sorted(set(y.tolist()), key=str)},
        "n_features_all": len(names),
        "feature_names": names,
        "tasks": {},
    }

    tasks = {
        "finegrained": y,
        "stable5": None,
        "coarse6": collapse_transients(y),
        "binary_transient": binary_transient(y),
    }
    stable_mask = np.array([not is_transition(yi) for yi in y])

    for abl_name, fams in ABLATIONS.items():
        for task_name, y_task in tasks.items():
            if task_name == "stable5":
                Xs, fn = _select(X[stable_mask], names, fams)
                ys = y[stable_mask]
                gs = groups[stable_mask]
                permute = 4 if abl_name in {"all", "baseline", "physical", "nonphysical"} else 0
                report = grouped_cv(
                    Xs, ys, gs, model="xgb", feature_names=fn, permute_repeats=permute
                )
            else:
                permute = 4 if (abl_name == "all" and task_name == "finegrained") else 0
                report, fn = _run_task(X, y_task, groups, names, fams, "xgb", permute)
            report["families"] = list(fams)
            report["n_features"] = len(fn)
            report["feature_names_used"] = fn
            summary["tasks"].setdefault(task_name, {})[abl_name] = {
                k: report[k]
                for k in report
                if k not in {"folds", "classification_report"}
            }
            summary["tasks"][task_name][abl_name]["classification_report"] = report.get(
                "classification_report"
            )
            summary["tasks"][task_name][abl_name]["folds"] = report.get("folds")
            print(
                f"w={window} {task_name:16s} {abl_name:20s}  "
                f"macroF1={report.get('macro_f1', float('nan')):.3f}  "
                f"n={report.get('n', 0)}",
                flush=True,
            )

    # Linear baseline on the winning-sized all-features stable task.
    Xs, fn = _select(X[stable_mask], names, ("baseline",))
    summary["tasks"]["stable5"]["baseline_logreg"] = grouped_cv(
        Xs, y[stable_mask], groups[stable_mask], model="logreg", feature_names=fn
    )
    Xs, fn = _select(X[stable_mask], names, FAMILIES)
    summary["tasks"]["stable5"]["all_logreg"] = grouped_cv(
        Xs, y[stable_mask], groups[stable_mask], model="logreg", feature_names=fn
    )

    out = OUT / f"feature_ablation_w{window}_h{hop}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")
    return summary


def run_inferred_config() -> dict:
    """All-10-video grouped CV on inverse-fitted Couzin weights (annotated_config.csv)."""
    import pandas as pd

    path = ROOT / "annotated_config.csv"
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in {"dataset", "start_frame", "end_frame", "fps", "label"}]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["label"].astype(str).to_numpy()
    groups = df["dataset"].astype(str).str.zfill(4).to_numpy()
    # Some dumps store dataset as int 66 not '0066'
    groups = np.array([g.zfill(4) if g.isdigit() else g for g in groups], dtype=object)
    report = grouped_cv(X, y, groups, model="xgb", feature_names=feat_cols, permute_repeats=6)
    out = {
        "n": int(len(y)),
        "n_videos": int(len(set(groups.tolist()))),
        "videos": sorted(set(groups.tolist()), key=str),
        "features": feat_cols,
        "label_counts": {lab: int(np.sum(y == lab)) for lab in sorted(set(y.tolist()))},
        "note": "15-frame (or dump-native) windows of STABLE labels only; inferred interaction parameters.",
        "xgb": {k: report[k] for k in report if k != "folds"},
        "folds": report.get("folds"),
    }
    # Compare against 3-Φ segment means in annotated_features.csv if aligned poorly —
    # those are whole-segment, so we evaluate them separately.
    feat_path = ROOT / "annotated_features.csv"
    if feat_path.exists():
        af = pd.read_csv(feat_path)
        phi_cols = [c for c in ("phi_trans", "phi_tan", "phi_rad_pm") if c in af.columns]
        if phi_cols and "label" in af.columns:
            y2 = af["label"].astype(str).to_numpy()
            g2 = af["dataset"].astype(str).to_numpy()
            g2 = np.array([g.zfill(4) if g.isdigit() else g for g in g2], dtype=object)
            X2 = af[phi_cols].to_numpy(dtype=np.float64)
            r2 = grouped_cv(X2, y2, g2, model="xgb", feature_names=phi_cols)
            out["segment_level_3phi"] = {k: r2[k] for k in r2 if k != "folds"}
    (OUT / "inferred_config_cv.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"inferred Couzin-weight grouped CV macroF1={report.get('macro_f1', float('nan')):.3f}  "
        f"n={len(y)} videos={out['n_videos']}"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--hop", type=int, default=5)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--inferred-only", action="store_true")
    p.add_argument("--skip-inferred", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.inferred_only:
        run(args.window, args.hop, rebuild=args.rebuild)
    if not args.skip_inferred:
        run_inferred_config()


if __name__ == "__main__":
    main()
