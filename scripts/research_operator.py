#!/usr/bin/env python3
"""Operator-theoretic LOVO vs 4 Φ and full physical, three temporal scales.

Scales
------
segment  — one sample per official annotated clip (591 H5 segments)
w15      — 15-frame windows, hop 5 (same rows as largen cache)
w5       — 5-frame windows, hop 5 (same rows as largen cache)

Caches features in gitignored research/cache/. Does not edit annotations.
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
from src.classify.sequences import SupervisedHMM, decoded_events, match_events, true_events
from src.features.dataset import load_positions_velocities, load_real_segments
from src.features.extended import (
    all_window_feature_names,
    columns_for_families,
    compute_extended_series,
    window_feature_dict,
)
from src.features.operator import (
    K_CANDIDATES,
    all_operator_names,
    columns_for_operator_families,
    window_operator_features,
)
import src.features.operator as operator_mod
from src.labels import CANONICAL, is_transition

OUT = ROOT / "research" / "results"
CACHE = ROOT / "research" / "cache"
FIG = ROOT / "research" / "figures"
TOL = 45

PHYS_WIN = lambda w, h: CACHE / f"largen_windows_w{w}_h{h}.npz"
OP_WIN = lambda w, h: CACHE / f"operator_windows_w{w}_h{h}.npz"
SEG_CACHE = CACHE / "operator_segments_h5.npz"

ABLATIONS = {
    "spatial": ("spatial",),
    "magnetic": ("magnetic",),
    "hodge": ("hodge",),
    "dmd": ("dmd",),
    "koopman": ("koopman",),
    "dyn": ("dyn",),
    "operator": ("operator",),
}

PHYS_ABLS = {
    "phi4": ("baseline",),
    "physical": ("baseline", "fluct", "rate", "shape", "tunstrom", "fields", "early"),
}

SCALES = {
    "segment": {"kind": "segment", "window": None, "hop": None},
    "w15": {"kind": "window", "window": 15, "hop": 5},
    "w5": {"kind": "window", "window": 5, "hop": 5},
}


def _group_label(n_fish: np.ndarray) -> np.ndarray:
    return np.array(["150_fish" if int(n) >= 140 else f"{int(n)}_fish" for n in n_fish], dtype=object)


def _slim(report: dict) -> dict:
    keep = (
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
        "permutation_importance",
        "n_features",
        "feature_names_used",
        "stable",
        "transient",
    )
    return {k: report[k] for k in keep if k in report}


def extract_windows(window: int, hop: int, rebuild: bool = False):
    path = OP_WIN(window, hop)
    phys_path = PHYS_WIN(window, hop)
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return z["X"], [str(n) for n in z["names"]], z["y"], z["groups"], z["n_fish"], z["meta"]
    if not phys_path.exists():
        raise FileNotFoundError(f"Need {phys_path}; run research_largen.py first")
    pz = np.load(phys_path, allow_pickle=True)
    meta = pz["meta"]
    y = pz["y"]
    groups = pz["groups"]
    n_fish = pz["n_fish"]
    names = all_operator_names()
    by_ds = defaultdict(list)
    for i, row in enumerate(meta):
        by_ds[str(row[0])].append(i)
    segs = load_real_segments()
    csv_of = {}
    for s in segs:
        csv_of.setdefault(s["dataset"], s["csv"])
    X = np.full((len(meta), len(names)), np.nan, dtype=np.float64)
    for ds, idxs in sorted(by_ds.items()):
        pos, vel = load_positions_velocities(ROOT / csv_of[ds])
        print(f"operator w={window} {ds} T={pos.shape[0]} N={pos.shape[1]} nwin={len(idxs)}", flush=True)
        cache: dict[int, dict] = {}
        for j, i in enumerate(idxs):
            a, b = int(meta[i][3]), int(meta[i][4])
            feat = window_operator_features(pos, vel, a, b, cache=cache, k=operator_mod.K_NN)
            X[i] = [feat.get(n, np.nan) for n in names]
            if (j + 1) % 2000 == 0:
                print(f"  {j+1}/{len(idxs)}", flush=True)
        print(f"  done {ds} cache_frames={len(cache)}", flush=True)
        cache.clear()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        groups=groups,
        n_fish=n_fish,
        names=np.array(names, dtype=object),
        meta=meta,
    )
    return X, names, y, groups, n_fish, meta


def extract_segments(rebuild: bool = False):
    path = SEG_CACHE
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return (
            z["Xop"],
            [str(n) for n in z["op_names"]],
            z["Xph"],
            [str(n) for n in z["ph_names"]],
            z["y"],
            z["groups"],
            z["n_fish"],
            z["meta"],
        )
    segs = load_real_segments()
    if not segs:
        raise RuntimeError("No H5 segments")
    op_names = all_operator_names()
    ph_names = all_window_feature_names()
    by_csv = defaultdict(list)
    for s in segs:
        by_csv[s["csv"]].append(s)
    rows_op, rows_ph, ys, gs, nf, meta = [], [], [], [], [], []
    for csv, slist in sorted(by_csv.items()):
        pos, vel = load_positions_velocities(ROOT / csv)
        print(
            f"segment {slist[0]['dataset']} T={pos.shape[0]} N={pos.shape[1]} nseg={len(slist)}",
            flush=True,
        )
        series = compute_extended_series(pos, vel)
        cache: dict[int, dict] = {}
        n_fish = int(pos.shape[1])
        for k, s in enumerate(slist):
            a, b = int(s["start"]), min(int(s["end"]), pos.shape[0])
            if b - a < 2:
                continue
            ofeat = window_operator_features(pos, vel, a, b, cache=cache, k=operator_mod.K_NN)
            pfeat = window_feature_dict(series, pos, vel, a, b)
            rows_op.append([ofeat.get(n, np.nan) for n in op_names])
            rows_ph.append([pfeat.get(n, np.nan) for n in ph_names])
            ys.append(s["label"])
            gs.append(s["dataset"])
            nf.append(n_fish)
            meta.append((s["dataset"], s["group"], s["label"], a, b))
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(slist)}", flush=True)
        print(f"  done {slist[0]['dataset']} n={len(slist)}", flush=True)
        cache.clear()
    Xop = np.asarray(rows_op, dtype=np.float64)
    Xph = np.asarray(rows_ph, dtype=np.float64)
    y = np.array(ys, dtype=object)
    groups = np.array(gs, dtype=object)
    n_fish = np.asarray(nf, dtype=np.int32)
    meta_arr = np.array(meta, dtype=object)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Xop=Xop,
        Xph=Xph,
        y=y,
        groups=groups,
        n_fish=n_fish,
        op_names=np.array(op_names, dtype=object),
        ph_names=np.array(ph_names, dtype=object),
        meta=meta_arr,
    )
    return Xop, op_names, Xph, ph_names, y, groups, n_fish, meta_arr


def load_scale(scale: str, rebuild: bool = False):
    spec = SCALES[scale]
    if spec["kind"] == "segment":
        Xop, op_names, Xph, ph_names, y, groups, n_fish, meta = extract_segments(rebuild=rebuild)
        return Xop, op_names, Xph, ph_names, y, groups, n_fish, meta
    Xop, op_names, y, groups, n_fish, meta = extract_windows(spec["window"], spec["hop"], rebuild=rebuild)
    pz = np.load(PHYS_WIN(spec["window"], spec["hop"]), allow_pickle=True)
    Xph, ph_names = pz["X"], [str(n) for n in pz["names"]]
    return Xop, op_names, Xph, ph_names, y, groups, n_fish, meta


def _run_tasks(X, y, groups, cols, fnames, model="xgb", permute=0, task_names=None) -> dict:
    n_videos = int(len(set(groups.tolist())))
    block = {"n": int(len(y)), "n_videos": n_videos, "tasks": {}}
    if n_videos < 2 or len(y) < 16:
        block["skipped"] = "too few videos"
        return block
    Xs = X[:, cols]
    stable = np.array([not is_transition(yi) for yi in y])
    tasks = {
        "stable5": (Xs[stable], y[stable], groups[stable]),
        "coarse6": (Xs, collapse_transients(y), groups),
        "binary_transient": (Xs, binary_transient(y), groups),
        "finegrained": (Xs, y, groups),
    }
    if task_names is not None:
        tasks = {t: tasks[t] for t in task_names if t in tasks}
    for task, (Xt, yt, gt) in tasks.items():
        if task == "stable5" and (len(yt) < 16 or len(set(gt.tolist())) < 2):
            continue
        # Singleton A→B labels break stratified internal splits in MLP/HGB.
        if task == "finegrained" and model != "xgb":
            continue
        do_perm = permute if task == "stable5" else 0
        rep = grouped_cv(
            Xt, yt, gt, model=model, feature_names=fnames, permute_repeats=do_perm
        )
        rep["n_features"] = len(fnames)
        rep["feature_names_used"] = list(fnames)
        block["tasks"][task] = _slim(rep)
        trans = None
        if task == "coarse6":
            trans = rep.get("per_class_f1", {}).get("transient")
        extra = f" trans={trans:.3f}" if trans is not None else ""
        print(
            f"  {task:16s} {model:5s} nf={len(fnames):3d}  "
            f"macroF1={rep.get('macro_f1', float('nan')):.3f}{extra}  n={rep.get('n', 0)}",
            flush=True,
        )
    return block


def _eval_split(Xop, y, groups, op_names, Xph, ph_names, *, permute: bool, models=("xgb",)) -> dict:
    out = {"n": int(len(y)), "n_videos": int(len(set(groups.tolist()))), "ablations": {}}
    for abl, fams in ABLATIONS.items():
        cols = columns_for_operator_families(fams, op_names)
        fn = [op_names[i] for i in cols]
        print(f" ablation {abl} ({len(fn)} feat)", flush=True)
        perm = 3 if permute and abl == "operator" else 0
        out["ablations"][abl] = _run_tasks(Xop, y, groups, cols, fn, model="xgb", permute=perm)
    for abl, fams in PHYS_ABLS.items():
        cols = columns_for_families(fams, ph_names)
        fn = [ph_names[i] for i in cols]
        print(f" ablation {abl} ({len(fn)} feat)", flush=True)
        out["ablations"][abl] = _run_tasks(Xph, y, groups, cols, fn, model="xgb", permute=0)
    cols_o = columns_for_operator_families(("operator",), op_names)
    cols_p = columns_for_families(
        ("baseline", "fluct", "rate", "shape", "tunstrom", "fields", "early"), ph_names
    )
    Xc = np.hstack([Xop[:, cols_o], Xph[:, cols_p]])
    fn = [op_names[i] for i in cols_o] + [ph_names[i] for i in cols_p]
    print(f" ablation operator+physical ({len(fn)} feat)", flush=True)
    out["ablations"]["operator+physical"] = _run_tasks(
        Xc, y, groups, list(range(len(fn))), fn, model="xgb", permute=0
    )
    cols = columns_for_operator_families(("operator",), op_names)
    fn = [op_names[i] for i in cols]
    out["classifiers"] = {}
    for model in models:
        print(f" classifier {model}", flush=True)
        out["classifiers"][model] = _run_tasks(Xop, y, groups, cols, fn, model=model, permute=0)
    return out


def _f1(block: dict, abl: str, task: str, key: str = "macro_f1"):
    rec = block.get("ablations", {}).get(abl, {}).get("tasks", {}).get(task, {})
    if key == "transient_f1":
        return rec.get("per_class_f1", {}).get("transient", np.nan)
    return rec.get(key, np.nan)


def _plot_scale(summary: dict, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    abls = [
        "phi4",
        "physical",
        "spatial",
        "magnetic",
        "hodge",
        "dmd",
        "koopman",
        "operator",
        "operator+physical",
    ]
    splits = ["pooled"] + [k for k, v in summary["by_group"].items() if "skipped" not in v]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(splits))
    width = 0.09
    cmap = plt.cm.tab10(np.linspace(0, 1, len(abls)))
    for i, abl in enumerate(abls):
        vals = []
        for sp in splits:
            block = summary["pooled"] if sp == "pooled" else summary["by_group"][sp]
            vals.append(_f1(block, abl, "stable5"))
        ax.bar(x + (i - len(abls) / 2) * width, vals, width, label=abl, color=cmap[i])
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("stable5 macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title(f"Operator-theoretic LOVO  {tag}")
    fig.tight_layout()
    fig.savefig(FIG / f"operator_stable5_{tag}.png", dpi=140)
    plt.close(fig)


def _plot_three_scales(combined: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    abls = [
        "phi4",
        "physical",
        "spatial",
        "magnetic",
        "hodge",
        "dmd",
        "koopman",
        "dyn",
        "operator",
        "operator+physical",
    ]
    scales = [s for s in ("segment", "w15", "w5") if s in combined]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(abls)))
    for ax, sc in zip(axes, scales):
        block = combined[sc]["pooled"]
        vals = [_f1(block, abl, "stable5") for abl in abls]
        ax.bar(np.arange(len(abls)), vals, color=cmap)
        ax.set_xticks(np.arange(len(abls)))
        ax.set_xticklabels(abls, rotation=55, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(sc)
        ax.axhline(0.8, color="0.6", ls="--", lw=0.8)
    axes[0].set_ylabel("pooled stable5 macro-F1")
    fig.suptitle("Operator vs 4 Φ vs physical — three temporal scales")
    fig.tight_layout()
    fig.savefig(FIG / "operator_stable5_three_scales.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    for ax, sc in zip(axes, scales):
        block = combined[sc]["pooled"]
        vals = [_f1(block, abl, "coarse6", "transient_f1") for abl in abls]
        ax.bar(np.arange(len(abls)), vals, color=cmap)
        ax.set_xticks(np.arange(len(abls)))
        ax.set_xticklabels(abls, rotation=55, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(sc)
    axes[0].set_ylabel("pooled coarse-6 transient F1")
    fig.suptitle("Transient superclass — three temporal scales")
    fig.tight_layout()
    fig.savefig(FIG / "operator_transient_three_scales.png", dpi=150)
    plt.close(fig)


def _spectrum_figure(X, names, y, tag: str, n_per=400) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = {n: i for i, n in enumerate(names)}
    stable = np.array([yi in CANONICAL for yi in y])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    rng = np.random.default_rng(0)
    pairs = [
        ("spat_knn_ell_mean", "hodge_rad_mean", "kNN edge length vs signed radial flux"),
        ("hodge_div_mean", "dmd_freq_max", "Signed Hodge divergence vs DMD frequency"),
        ("dyn_dlog_ell", "koop_resid_max", "Edge-length growth vs Koopman residual"),
    ]
    colors = {
        "traveling": "#1f77b4",
        "milling": "#d62728",
        "shoaling": "#2ca02c",
        "expansion": "#ff7f0e",
        "compaction": "#9467bd",
    }
    for ax, (a, b, title) in zip(axes, pairs):
        if a not in idx or b not in idx:
            ax.set_title(title + " (missing)")
            continue
        ia, ib = idx[a], idx[b]
        for lab in CANONICAL:
            m = stable & (y == lab)
            pts = np.flatnonzero(m)
            if pts.size == 0:
                continue
            take = rng.choice(pts, size=min(n_per, pts.size), replace=False)
            ax.scatter(X[take, ia], X[take, ib], s=6, alpha=0.35, c=colors[lab], label=lab, linewidths=0)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"operator_planes_{tag}.png", dpi=150)
    plt.close(fig)


def _residual_figure(X, names, y, groups, meta, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "koop_resid_max" not in names:
        return
    ir = names.index("koop_resid_max")
    segs = load_real_segments()
    by = defaultdict(list)
    for s in segs:
        by[s["dataset"]].append(s)
    # pick the video with the most official transients
    vid = max(by, key=lambda v: sum(1 for s in by[v] if is_transition(s["label"])))
    m = groups == vid
    order = np.argsort([int(meta[i][3]) for i in np.flatnonzero(m)])
    idx = np.flatnonzero(m)[order]
    t = np.array([(int(meta[i][3]) + int(meta[i][4])) // 2 for i in idx])
    r = X[idx, ir]
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    ax.plot(t, r, color="0.2", lw=0.8)
    ymax = np.nanmax(r[np.isfinite(r)]) if np.isfinite(r).any() else 1.0
    for s in by[vid]:
        if not is_transition(s["label"]):
            continue
        ax.axvspan(int(s["start"]), int(s["end"]), color="#d62728", alpha=0.18)
    ax.set_xlabel(f"frame ({vid})")
    ax.set_ylabel("Koopman residual (max)")
    ax.set_ylim(0, ymax * 1.05 if np.isfinite(ymax) else 1)
    ax.set_title(f"Koopman residual vs official transients — {vid} ({tag})")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"operator_residual_{tag}_{vid}.png", dpi=150)
    plt.close(fig)


def _events(X, names, y, groups, meta) -> dict:
    use = [
        "spat_knn_l2",
        "spat_knn_ell_mean",
        "mag_q50_dl2",
        "hodge_rad_mean",
        "hodge_div_mean",
        "dmd_growth_max",
        "dmd_freq_max",
        "koop_resid_mean",
        "dyn_dlog_ell",
    ]
    cols = [names.index(n) for n in use if n in names]
    segs = load_real_segments()
    by = defaultdict(list)
    for s in segs:
        by[s["dataset"]].append(s)
    videos = sorted(set(groups.tolist()), key=str)
    hmm_rows, iid_rows, res_rows = [], [], []
    from src.classify.grouped import _inv_weights, _xgb
    from sklearn.preprocessing import LabelEncoder

    for held in videos:
        tr = groups != held
        te = groups == held
        Xtr = X[tr][:, cols]
        ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        def _fill(M):
            M = M.copy()
            bad = ~np.isfinite(M)
            if bad.any():
                M[bad] = np.take(med, np.where(bad)[1])
            return M

        train_vids = sorted(set(groups[tr].tolist()), key=str)
        Xs, ys = [], []
        for vid in train_vids:
            m = groups == vid
            if m.sum() < 8:
                continue
            order = np.argsort([int(meta[i][3]) for i in np.flatnonzero(m)])
            ix = np.flatnonzero(m)[order]
            lab = np.array([str(y[i]) if y[i] in CANONICAL else "" for i in ix], dtype=object)
            Xs.append(_fill(X[ix][:, cols]))
            ys.append(lab)
        if len(Xs) < 2:
            continue
        hmm = SupervisedHMM()
        hmm.fit(Xs, ys)
        order = np.argsort([int(meta[i][3]) for i in np.flatnonzero(te)])
        ix = np.flatnonzero(te)[order]
        Xte = _fill(X[ix][:, cols])
        states = hmm.viterbi(Xte)
        t_mid = np.array([(int(meta[i][3]) + int(meta[i][4])) // 2 for i in ix])
        pred_e = decoded_events(states, min_run=2)
        for e in pred_e:
            e["t"] = int(t_mid[min(int(e["t"]), len(t_mid) - 1)])
        true_e = true_events(by[held], 10**9)
        hmm_rows.append(match_events(true_e, pred_e, TOL))
        sttr = np.array([yi in CANONICAL for yi in ytr])
        if sttr.sum() < 8:
            continue
        le = LabelEncoder()
        le.fit(list(CANONICAL))
        clf = _xgb(len(CANONICAL))
        yenc = le.transform(ytr[sttr])
        clf.fit(_fill(Xtr[sttr]), yenc, sample_weight=_inv_weights(yenc))
        pred = le.inverse_transform(np.clip(clf.predict(Xte), 0, 4))
        iid_e = decoded_events(pred, min_run=2)
        for e in iid_e:
            e["t"] = int(t_mid[min(int(e["t"]), len(t_mid) - 1)])
        iid_rows.append(match_events(true_e, iid_e, TOL))
        if "koop_resid_max" in names:
            r = X[ix, names.index("koop_resid_max")]
            r = np.nan_to_num(r, nan=np.nanmedian(r[np.isfinite(r)]) if np.isfinite(r).any() else 0.0)
            thr = np.nanmedian(r) + 1.5 * (np.nanstd(r) + 1e-6)
            peaks, on, t0 = [], False, 0
            for t, val in enumerate(r):
                if val > thr and not on:
                    on, t0 = True, t
                elif on and val <= thr:
                    peaks.append(t0)
                    on = False
            res_e = []
            for t in peaks:
                src = str(pred[max(0, t - 1)])
                dst = str(pred[min(len(pred) - 1, t + 1)])
                if src in CANONICAL and dst in CANONICAL and src != dst:
                    res_e.append({"src": src, "dst": dst, "t": int(t_mid[t])})
            res_rows.append(match_events(true_e, res_e, TOL))

    def _sum(rows):
        if not rows:
            return {}
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        idn = (
            np.nanmean([r["identity_accuracy"] for r in rows if r["tp"] > 0])
            if any(r["tp"] > 0 for r in rows)
            else 0.0
        )
        return {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "identity_accuracy": float(idn),
            "n_true": int(sum(r["n_true"] for r in rows)),
            "n_pred": int(sum(r["n_pred"] for r in rows)),
        }

    return {
        "hmm": _sum(hmm_rows),
        "iid_windows": _sum(iid_rows),
        "residual_cusum": _sum(res_rows),
        "tol_frames": TOL,
    }


def _plot_k_sweep(sweep: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    ks = [int(r["k"]) for r in sweep["pooled"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(ks, [r["stable5_macro_f1"] for r in sweep["pooled"]], "o-", label="pooled", lw=2)
    for split, color in (("10_fish", "#1f77b4"), ("30_fish", "#ff7f0e"), ("70_fish", "#2ca02c")):
        rows = sweep["by_group"].get(split, [])
        if rows:
            ax.plot([int(r["k"]) for r in rows], [r["stable5_macro_f1"] for r in rows], "s--", label=split, color=color)
    ax.axvline(sweep["k_star"], color="0.3", ls=":", label=f"frozen k={sweep['k_star']}")
    ax.set_xlabel("kNN degree k (capped at N−1)")
    ax.set_ylabel("stable5 LOVO macro-F1")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Unscaled kNN degree sweep — segment-level grouped LOVO")
    fig.tight_layout()
    fig.savefig(FIG / "operator_k_sweep.png", dpi=150)
    plt.close(fig)


def freeze_k(k_star: int) -> None:
    """Set runtime k and persist the frozen constant in operator.py."""
    operator_mod.K_NN = int(k_star)
    path = ROOT / "src" / "features" / "operator.py"
    text = path.read_text()
    old = None
    for line in text.splitlines():
        if line.startswith("K_NN =") and "CANDIDATES" not in line:
            old = line
            break
    if old is None:
        raise RuntimeError("could not find K_NN assignment")
    new = (
        f"K_NN = {int(k_star)}  "
        "# frozen from leakage-safe segment LOVO k-sweep; see research/results/operator_k_sweep.json"
    )
    path.write_text(text.replace(old, new, 1))
    print(f"froze K_NN = {k_star}", flush=True)


def sweep_k(rebuild: bool = False) -> dict:
    """Leakage-safe segment LOVO over k ∈ {3..9}; freeze argmax stable5 macro-F1."""
    cache_path = CACHE / "operator_ksweep_segments.npz"
    segs = load_real_segments()
    op_names = all_operator_names()
    candidates = list(K_CANDIDATES)
    if cache_path.exists() and not rebuild:
        z = np.load(cache_path, allow_pickle=True)
        y = z["y"]
        groups = z["groups"]
        n_fish = z["n_fish"]
        Xs = {int(k): z[f"Xk{k}"] for k in candidates}
        print(f"loaded k-sweep cache {cache_path} n={len(y)}", flush=True)
    else:
        by_csv = defaultdict(list)
        for s in segs:
            by_csv[s["csv"]].append(s)
        buckets = {k: [] for k in candidates}
        ys, gs, nf = [], [], []
        for csv, slist in sorted(by_csv.items()):
            pos, vel = load_positions_velocities(ROOT / csv)
            print(
                f"k-sweep {slist[0]['dataset']} T={pos.shape[0]} N={pos.shape[1]} nseg={len(slist)}",
                flush=True,
            )
            n = int(pos.shape[1])
            for k in candidates:
                cache: dict[int, dict] = {}
                rows = []
                for s in slist:
                    a, b = int(s["start"]), min(int(s["end"]), pos.shape[0])
                    if b - a < 2:
                        continue
                    feat = window_operator_features(pos, vel, a, b, cache=cache, k=k)
                    rows.append([feat.get(n_, np.nan) for n_ in op_names])
                    if k == candidates[0]:
                        ys.append(s["label"])
                        gs.append(s["dataset"])
                        nf.append(n)
                buckets[k].extend(rows)
                print(f"  k={k} done", flush=True)
        y = np.array(ys, dtype=object)
        groups = np.array(gs, dtype=object)
        n_fish = np.asarray(nf, dtype=np.int32)
        Xs = {k: np.asarray(buckets[k], dtype=np.float64) for k in candidates}
        CACHE.mkdir(parents=True, exist_ok=True)
        payload = {"y": y, "groups": groups, "n_fish": n_fish, "names": np.array(op_names, dtype=object)}
        for k, X in Xs.items():
            payload[f"Xk{k}"] = X
        np.savez_compressed(cache_path, **payload)

    cols = list(range(len(op_names)))
    fnames = list(op_names)
    gl = _group_label(n_fish)
    pooled_rows = []
    by_group = {g: [] for g in sorted(set(gl.tolist()), key=str)}
    for k in candidates:
        print(f"\n=== k-sweep k={k} pooled ===", flush=True)
        block = _run_tasks(
            Xs[k], y, groups, cols, fnames, model="xgb", permute=0, task_names=("stable5", "coarse6")
        )
        s5 = block.get("tasks", {}).get("stable5", {})
        c6 = block.get("tasks", {}).get("coarse6", {})
        pooled_rows.append(
            {
                "k": k,
                "stable5_macro_f1": s5.get("macro_f1", np.nan),
                "stable5_per_class_f1": s5.get("per_class_f1", {}),
                "coarse6_macro_f1": c6.get("macro_f1", np.nan),
                "transient_f1": (c6.get("per_class_f1") or {}).get("transient", np.nan),
                "n_stable": s5.get("n"),
            }
        )
        for g in by_group:
            m = gl == g
            print(f"  {g} k={k}", flush=True)
            gb = _run_tasks(
                Xs[k][m], y[m], groups[m], cols, fnames, model="xgb", permute=0, task_names=("stable5",)
            )
            gs5 = gb.get("tasks", {}).get("stable5", {})
            by_group[g].append({"k": k, "stable5_macro_f1": gs5.get("macro_f1", np.nan), "n_stable": gs5.get("n")})

    def _key(r):
        v = r["stable5_macro_f1"]
        return -1.0 if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)

    k_star = int(max(pooled_rows, key=_key)["k"])

    # Nested LOVO: pick k on the 9 training videos, score the held-out video.
    videos = sorted(set(groups.tolist()), key=str)
    nested = []
    for held in videos:
        inner = []
        tr = groups != held
        for k in candidates:
            ib = _run_tasks(
                Xs[k][tr], y[tr], groups[tr], cols, fnames, model="xgb", permute=0, task_names=("stable5",)
            )
            inner.append({"k": k, "stable5_macro_f1": ib.get("tasks", {}).get("stable5", {}).get("macro_f1", np.nan)})
        k_inner = int(max(inner, key=_key)["k"])
        te = groups == held
        # single-video outer score: train on the 9, test on held (grouped_cv with 2 groups is wrong).
        from src.classify.grouped import _inv_weights, _xgb
        from sklearn.preprocessing import LabelEncoder

        st_tr = np.array([yi in CANONICAL for yi in y[tr]])
        st_te = np.array([yi in CANONICAL for yi in y[te]])
        outer_f1 = np.nan
        if st_tr.sum() >= 16 and st_te.sum() >= 1:
            Xtr, Xte = Xs[k_inner][tr][st_tr], Xs[k_inner][te][st_te]
            ytr, yte = y[tr][st_tr], y[te][st_te]
            med = np.nanmedian(Xtr, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            for mat in (Xtr, Xte):
                bad = ~np.isfinite(mat)
                if bad.any():
                    mat[bad] = np.take(med, np.where(bad)[1])
            present = [c for c in CANONICAL if (ytr == c).any()]
            if len(present) >= 2:
                le = LabelEncoder()
                le.fit(present)
                known = np.array([yi in le.classes_ for yi in yte])
                clf = _xgb(len(le.classes_))
                yenc = le.transform(ytr)
                clf.fit(Xtr, yenc, sample_weight=_inv_weights(yenc))
                pred = np.array([""] * len(yte), dtype=object)
                pe = clf.predict(Xte)
                pred[known] = le.inverse_transform(np.clip(pe[known], 0, len(le.classes_) - 1))
                from sklearn.metrics import f1_score

                outer_f1 = float(
                    f1_score(yte, pred, average="macro", labels=list(CANONICAL), zero_division=0)
                )
        nested.append({"held_out": str(held), "k_inner": k_inner, "inner": inner, "outer_stable5_macro_f1": outer_f1})
        print(f" nested held={held} k*={k_inner} outerF1={outer_f1:.3f}", flush=True)

    sweep = {
        "candidates": candidates,
        "k_star": k_star,
        "selection": "argmax pooled segment-level grouped LOVO stable5 macro-F1 on the full operator suite (XGB)",
        "n_segments": int(len(y)),
        "n_videos": int(len(videos)),
        "pooled": pooled_rows,
        "by_group": by_group,
        "nested_lovo": nested,
        "nested_k_counts": {str(k): int(sum(1 for r in nested if r["k_inner"] == k)) for k in candidates},
        "note": "k capped at N-1 per frame (10-fish: k=9 is a 9-NN graph). Unscaled kNN, no Delaunay, no NND σ.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "operator_k_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")
    _plot_k_sweep(sweep)
    freeze_k(k_star)
    print(f"k-sweep complete: k*={k_star}", flush=True)
    return sweep


def run_scale(scale: str, rebuild: bool = False) -> dict:
    spec = SCALES[scale]
    Xop, op_names, Xph, ph_names, y, groups, n_fish, meta = load_scale(scale, rebuild=rebuild)
    gl = _group_label(n_fish)
    tag = scale if scale == "segment" else f"w{spec['window']}_h{spec['hop']}"
    summary = {
        "scale": scale,
        "window": spec["window"],
        "hop": spec["hop"],
        "n_rows": int(len(y)),
        "n_videos": int(len(set(groups.tolist()))),
        "videos": sorted(set(groups.tolist()), key=str),
        "n_operator_features": len(op_names),
        "operator_names": op_names,
        "note": (
            "H5 trajectories. Operator features use no Φ / M / dlogR. "
            "150-fish skipped per-N. Same windows as largen_lovo when scale is w5/w15."
        ),
    }
    print(f"\n=== pooled operator {tag} n={len(y)} ===", flush=True)
    summary["pooled"] = _eval_split(
        Xop, y, groups, op_names, Xph, ph_names, permute=True, models=("xgb", "logreg", "hgb")
    )
    summary["by_group"] = {}
    for g in sorted(set(gl.tolist()), key=str):
        m = gl == g
        print(f"\n=== {tag} {g} n={int(m.sum())} ===", flush=True)
        summary["by_group"][g] = _eval_split(
            Xop[m], y[m], groups[m], op_names, Xph[m], ph_names, permute=False, models=("xgb",)
        )
    if scale != "segment":
        print("events…", flush=True)
        summary["events"] = _events(Xop, op_names, y, groups, meta)
        _residual_figure(Xop, op_names, y, groups, meta, tag)
    _plot_scale(summary, tag)
    _spectrum_figure(Xop, op_names, y, tag)
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / f"operator_lovo_{tag}.json"
    outp.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", outp, flush=True)
    return summary


def _table_row(combined: dict, split: str, task: str, key: str = "macro_f1") -> dict:
    row = {}
    for sc, summary in combined.items():
        block = summary["pooled"] if split == "pooled" else summary.get("by_group", {}).get(split, {})
        for abl in list(ABLATIONS) + ["phi4", "physical", "operator+physical"]:
            row[f"{sc}:{abl}"] = _f1(block, abl, task, key)
        clfs = (summary["pooled"] if split == "pooled" else block).get("classifiers", {})
        for model, rec in clfs.items():
            row[f"{sc}:clf_{model}"] = rec.get("tasks", {}).get(task, {}).get("macro_f1", np.nan)
    return row


def combine(summaries: dict) -> dict:
    combined = {k: v for k, v in summaries.items()}
    tables = {}
    for task in ("stable5", "coarse6", "binary_transient", "finegrained"):
        tables[task] = {}
        for split in ["pooled", "10_fish", "30_fish", "70_fish"]:
            tables[task][split] = _table_row(combined, split, task)
            if task == "coarse6":
                tables[task][f"{split}_transient_f1"] = _table_row(combined, split, task, "transient_f1")
    out = {
        "scales": {k: {"n_rows": v.get("n_rows"), "n_videos": v.get("n_videos")} for k, v in combined.items()},
        "tables": tables,
        "events": {k: v.get("events", {}) for k, v in combined.items()},
        "by_scale": combined,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "operator_lovo_three_scales.json").write_text(json.dumps(out, indent=2) + "\n")
    _plot_three_scales(combined)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["segment", "w15", "w5", "all", "ksweep"], default="all")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--skip-ksweep", action="store_true")
    args = p.parse_args()
    if args.scale == "ksweep" or not args.skip_ksweep:
        sweep_k(rebuild=args.rebuild)
        if args.scale == "ksweep":
            return
    scales = ["segment", "w15", "w5"] if args.scale == "all" else [args.scale]
    summaries = {}
    for sc in scales:
        summaries[sc] = run_scale(sc, rebuild=args.rebuild)
    if len(summaries) > 1:
        combine(summaries)
    elif (OUT / "operator_lovo_segment.json").exists() and (OUT / "operator_lovo_w15_h5.json").exists() and (
        OUT / "operator_lovo_w5_h5.json"
    ).exists():
        loaded = {}
        for sc, fn in (
            ("segment", "operator_lovo_segment.json"),
            ("w15", "operator_lovo_w15_h5.json"),
            ("w5", "operator_lovo_w5_h5.json"),
        ):
            loaded[sc] = json.loads((OUT / fn).read_text())
        combine(loaded)


if __name__ == "__main__":
    main()
