#!/usr/bin/env python3
"""Large-N 5-frame and 15-frame leave-one-video-out feature evaluation.

Uses on-disk H5 trajectories (gitignored). Annotations are not edited.
150-fish is a single video, so per-N LOVO is skipped; it is included in pooled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.grouped import binary_transient, collapse_transients, grouped_cv
from src.features.dataset import load_positions_velocities, load_real_segments
from src.features.extended import (
    all_window_feature_names,
    columns_for_families,
    compute_extended_series,
    iter_bounds,
    window_feature_dict,
)
from src.labels import is_transition

OUT = ROOT / "research" / "results"
CACHE = ROOT / "research" / "cache"
FIG = ROOT / "research" / "figures"

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
    "baseline+rate": ("baseline", "rate"),
    "baseline+spectral": ("baseline", "spectral"),
    "baseline+corr": ("baseline", "corr"),
    "baseline+fields": ("baseline", "fields"),
    "physical": ("baseline", "fluct", "rate", "shape", "tunstrom", "fields", "early"),
    "nonphysical": ("spectral", "corr", "info"),
    "all": FAMILIES,
}

PERMUTE_ABLS = {"baseline", "baseline+rate", "physical", "nonphysical", "all"}


def _cache_path(window: int, hop: int) -> Path:
    return CACHE / f"largen_windows_w{window}_h{hop}.npz"


def _select(X, names, families):
    cols = columns_for_families(families, names)
    return X[:, cols], [names[i] for i in cols]


def build_or_load(window: int, hop: int, rebuild: bool = False):
    path = _cache_path(window, hop)
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return (
            z["X"],
            z["y"],
            z["groups"],
            z["n_fish"],
            [str(n) for n in z["names"]],
            z["meta"],
        )
    segs = load_real_segments()
    if not segs:
        raise RuntimeError("No real segments with trajectories on disk")
    names = all_window_feature_names()
    xs, ys, gs, nf, meta = [], [], [], [], []
    by_csv = defaultdict(list)
    for s in segs:
        by_csv[s["csv"]].append(s)
    for csv, slist in sorted(by_csv.items()):
        pos, vel = load_positions_velocities(ROOT / csv)
        print(
            f"series {slist[0]['dataset']} T={pos.shape[0]} N={pos.shape[1]}",
            flush=True,
        )
        series = compute_extended_series(pos, vel)
        n_fish = int(pos.shape[1])
        for s in slist:
            a0, b0 = s["start"], min(s["end"], pos.shape[0])
            if b0 - a0 < window:
                continue
            bounds = iter_bounds(b0 - a0, window, hop)
            for w0, w1 in bounds:
                feat = window_feature_dict(series, pos, vel, a0 + w0, a0 + w1)
                xs.append([feat.get(n, np.nan) for n in names])
                ys.append(s["label"])
                gs.append(s["dataset"])
                nf.append(n_fish)
                meta.append((s["dataset"], s["group"], s["label"], a0 + w0, a0 + w1))
        print(f"  windows so far {len(xs)}", flush=True)
    X = np.vstack(xs) if xs else np.zeros((0, len(names)))
    y = np.array(ys, dtype=object)
    groups = np.array(gs, dtype=object)
    n_fish_arr = np.asarray(nf, dtype=np.int32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        groups=groups,
        n_fish=n_fish_arr,
        names=np.array(names, dtype=object),
        meta=np.array(meta, dtype=object),
    )
    return X, y, groups, n_fish_arr, names, np.array(meta, dtype=object)


def _slim(report: dict) -> dict:
    keep = {
        "model",
        "n",
        "n_groups",
        "groups",
        "labels",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "per_class_f1",
        "label_counts",
        "confusion_matrix",
        "permutation_importance",
        "families",
        "n_features",
        "feature_names_used",
        "stable",
        "transient",
        "classification_report",
    }
    return {k: report[k] for k in keep if k in report}


def _eval_block(X, y, groups, names, *, permute: bool) -> dict:
    n_videos = int(len(set(groups.tolist())))
    block = {
        "n": int(len(y)),
        "n_videos": n_videos,
        "videos": sorted(set(groups.tolist()), key=str),
        "label_counts": {lab: int(np.sum(y == lab)) for lab in sorted(set(y.tolist()), key=str)},
        "tasks": {},
    }
    if n_videos < 2 or len(y) < 16:
        block["skipped"] = "too few videos or rows for LOVO"
        return block
    stable_mask = np.array([not is_transition(yi) for yi in y])
    tasks = {
        "stable5": None,
        "coarse6": collapse_transients(y),
        "binary_transient": binary_transient(y),
        "finegrained": y,
    }
    for abl, fams in ABLATIONS.items():
        for task, y_task in tasks.items():
            do_perm = int(permute and abl in PERMUTE_ABLS and task == "stable5")
            if task == "stable5":
                if stable_mask.sum() < 16 or len(set(groups[stable_mask].tolist())) < 2:
                    continue
                Xs, fn = _select(X[stable_mask], names, fams)
                report = grouped_cv(
                    Xs,
                    y[stable_mask],
                    groups[stable_mask],
                    model="xgb",
                    feature_names=fn,
                    permute_repeats=4 if do_perm else 0,
                )
            else:
                Xs, fn = _select(X, names, fams)
                report = grouped_cv(
                    Xs, y_task, groups, model="xgb", feature_names=fn, permute_repeats=0
                )
            report["families"] = list(fams)
            report["n_features"] = len(fn)
            report["feature_names_used"] = fn
            block["tasks"].setdefault(task, {})[abl] = _slim(report)
            trans_f1 = None
            if task == "coarse6":
                trans_f1 = report.get("per_class_f1", {}).get("transient")
            print(
                f"  {task:16s} {abl:22s}  macroF1={report.get('macro_f1', float('nan')):.3f}"
                + (f"  transF1={trans_f1:.3f}" if trans_f1 is not None else "")
                + f"  n={report.get('n', 0)}",
                flush=True,
            )
    return block


def _plot_stable(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["pooled"] + [k for k in summary["by_group"] if "skipped" not in summary["by_group"][k]]
    abls = ["baseline", "baseline+rate", "baseline+spectral", "physical", "nonphysical", "all"]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(splits))
    width = 0.13
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    for i, abl in enumerate(abls):
        vals = []
        for sp in splits:
            block = summary["pooled"] if sp == "pooled" else summary["by_group"][sp]
            rec = block.get("tasks", {}).get("stable5", {}).get(abl, {})
            vals.append(rec.get("macro_f1", np.nan))
        ax.bar(x + (i - 2.5) * width, vals, width, label=abl, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("stable5 macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title(f"Large-N LOVO  w={summary['window']} hop={summary['hop']}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(window: int, hop: int, rebuild: bool = False) -> dict:
    X, y, groups, n_fish, names, _meta = build_or_load(window, hop, rebuild=rebuild)
    # Video 0200 is stored as N=151.
    group_label = np.array(
        ["150_fish" if int(n) >= 140 else f"{int(n)}_fish" for n in n_fish],
        dtype=object,
    )
    summary = {
        "window": window,
        "hop": hop,
        "n_windows": int(len(y)),
        "n_videos": int(len(set(groups.tolist()))),
        "videos": sorted(set(groups.tolist()), key=str),
        "n_features": len(names),
        "feature_names": names,
        "label_counts": {lab: int(np.sum(y == lab)) for lab in sorted(set(y.tolist()), key=str)},
        "by_group_n": {
            g: int(np.sum(group_label == g)) for g in sorted(set(group_label.tolist()), key=str)
        },
        "note": "5/15-frame windows on official segments. H5s from HF bucket josefberman/SchoolingFish. Annotations not edited. 150-fish is one video so per-N LOVO is skipped.",
    }
    print(f"\n=== pooled w={window} n={len(y)} videos={summary['n_videos']} ===", flush=True)
    summary["pooled"] = _eval_block(X, y, groups, names, permute=True)
    summary["by_group"] = {}
    for g in sorted(set(group_label.tolist()), key=str):
        m = group_label == g
        print(f"\n=== {g} n={int(m.sum())} ===", flush=True)
        summary["by_group"][g] = _eval_block(X[m], y[m], groups[m], names, permute=False)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"largen_lovo_w{window}_h{hop}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    FIG.mkdir(parents=True, exist_ok=True)
    _plot_stable(summary, FIG / f"largen_stable5_w{window}_h{hop}.png")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--hop", type=int, default=5)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    run(args.window, args.hop, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
