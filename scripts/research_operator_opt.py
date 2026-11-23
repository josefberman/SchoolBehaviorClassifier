#!/usr/bin/env python3
"""Iteration 5: optimized operator families + hyp sweep + three-scale LOVO.

Does not edit annotations. Caches in gitignored research/cache/.
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
    FAMILY_NAMES,
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
SEED = 42

PHYS_WIN = lambda w, h: CACHE / f"largen_windows_w{w}_h{h}.npz"
OP_WIN = lambda w, h: CACHE / f"operator_opt_windows_w{w}_h{h}.npz"
SEG_CACHE = CACHE / "operator_opt_segments_h5.npz"

PHYS_ABLS = {
    "phi4": ("baseline",),
    "physical": ("baseline", "fluct", "rate", "shape", "tunstrom", "fields", "early"),
}

STRUCT_ABLS = {
    "operator_v4": ("operator_v4",),
    "connection": ("connection",),
    "hodge2": ("hodge2",),
    "kedmd": ("kedmd",),
    "hankel": ("hankel",),
    "v4+connection": ("operator_v4", "connection"),
    "v4+hodge2": ("operator_v4", "hodge2"),
    "v4+kedmd": ("operator_v4", "kedmd"),
    "operator_opt": ("operator",),
}

KEDMD_CFGS = [(0.5, 4, "s05", "r4"), (0.5, 8, "s05", "r8"), (1.0, 4, "s1", "r4"), (1.0, 8, "s1", "r8"), (2.0, 4, "s2", "r4"), (2.0, 8, "s2", "r8")]
HANKEL_CFGS = [(3, 4), (3, 8), (6, 4), (6, 8)]


def _group_label(n_fish):
    return np.array(["150_fish" if int(n) >= 140 else f"{int(n)}_fish" for n in n_fish], dtype=object)


def _slim(report):
    keep = (
        "model",
        "n",
        "n_groups",
        "macro_f1",
        "balanced_accuracy",
        "per_class_f1",
        "permutation_importance",
        "shap_importance",
        "n_features",
        "feature_names_used",
        "labels",
    )
    return {k: report[k] for k in keep if k in report}


def _stable5(X, y, groups, cols, fnames, permute=0, shap=False, model="xgb"):
    m = np.array([not is_transition(yi) for yi in y])
    if m.sum() < 16 or len(set(groups[m].tolist())) < 2:
        return {}
    return grouped_cv(
        X[m][:, cols],
        y[m],
        groups[m],
        model=model,
        feature_names=fnames,
        permute_repeats=permute,
        shap=shap,
    )


def extract_segments(k: int, rebuild=False):
    path = CACHE / f"operator_opt_segments_k{k}.npz"
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return z["Xop"], [str(n) for n in z["op_names"]], z["Xph"], [str(n) for n in z["ph_names"]], z["y"], z["groups"], z["n_fish"], z["meta"]
    segs = load_real_segments()
    op_names = all_operator_names()
    ph_names = all_window_feature_names()
    by_csv = defaultdict(list)
    for s in segs:
        by_csv[s["csv"]].append(s)
    rows_op, rows_ph, ys, gs, nf, meta = [], [], [], [], [], []
    operator_mod.K_NN = int(k)
    for csv, slist in sorted(by_csv.items()):
        pos, vel = load_positions_velocities(ROOT / csv)
        print(f"opt-seg k={k} {slist[0]['dataset']} N={pos.shape[1]} nseg={len(slist)}", flush=True)
        series = compute_extended_series(pos, vel)
        cache = {}
        n_fish = int(pos.shape[1])
        for s in slist:
            a, b = int(s["start"]), min(int(s["end"]), pos.shape[0])
            if b - a < 2:
                continue
            ofeat = window_operator_features(pos, vel, a, b, cache=cache, k=k)
            pfeat = window_feature_dict(series, pos, vel, a, b)
            rows_op.append([ofeat.get(n, np.nan) for n in op_names])
            rows_ph.append([pfeat.get(n, np.nan) for n in ph_names])
            ys.append(s["label"])
            gs.append(s["dataset"])
            nf.append(n_fish)
            meta.append((s["dataset"], s["group"], s["label"], a, b))
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
        k=k,
    )
    return Xop, op_names, Xph, ph_names, y, groups, n_fish, meta_arr


def extract_windows(window, hop, k, rebuild=False):
    path = CACHE / f"operator_opt_windows_w{window}_h{hop}_k{k}.npz"
    phys_path = PHYS_WIN(window, hop)
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return z["X"], [str(n) for n in z["names"]], z["y"], z["groups"], z["n_fish"], z["meta"]
    pz = np.load(phys_path, allow_pickle=True)
    meta, y, groups, n_fish = pz["meta"], pz["y"], pz["groups"], pz["n_fish"]
    names = all_operator_names()
    by_ds = defaultdict(list)
    for i, row in enumerate(meta):
        by_ds[str(row[0])].append(i)
    segs = load_real_segments()
    csv_of = {}
    for s in segs:
        csv_of.setdefault(s["dataset"], s["csv"])
    X = np.full((len(meta), len(names)), np.nan)
    operator_mod.K_NN = int(k)
    for ds, idxs in sorted(by_ds.items()):
        pos, vel = load_positions_velocities(ROOT / csv_of[ds])
        print(f"opt-win w={window} k={k} {ds} N={pos.shape[1]} n={len(idxs)}", flush=True)
        cache = {}
        for j, i in enumerate(idxs):
            a, b = int(meta[i][3]), int(meta[i][4])
            feat = window_operator_features(pos, vel, a, b, cache=cache, k=k)
            X[i] = [feat.get(n, np.nan) for n in names]
            if (j + 1) % 2000 == 0:
                print(f"  {j+1}/{len(idxs)}", flush=True)
        cache.clear()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, groups=groups, n_fish=n_fish, names=np.array(names, dtype=object), meta=meta)
    return X, names, y, groups, n_fish, meta


def _kedmd_cols(sm_tag, r_tag, names):
    pref = f"kedmd_{sm_tag}_{r_tag}_"
    return [i for i, n in enumerate(names) if n.startswith(pref) or n == "kedmd_vamp2"]


def _hankel_cols(delays, rank, names):
    pref = f"hdmd_d{delays}_r{rank}_"
    return [i for i, n in enumerate(names) if n.startswith(pref)]


def frozen_cols(hyps, names):
    """Column indices for the frozen hyp selection (not the kitchen-sink vector)."""
    sel = str(hyps.get("selection", "operator_opt_all"))
    if hyps.get("use_all_families") or sel in {"operator_opt_all", "operator"}:
        return columns_for_operator_families(("operator",), names)
    if sel == "operator_v4":
        return columns_for_operator_families(("operator_v4",), names)
    base = columns_for_operator_families(("operator_v4", "connection", "hodge2"), names)
    if sel.startswith("base+kedmd_"):
        st, rt = sel.split("base+kedmd_")[-1].split("_")  # s05_r4
        return sorted(set(base + _kedmd_cols(st, rt, names)))
    if sel.startswith("base+hdmd_"):
        dtag, rtag = sel.split("base+hdmd_")[-1].split("_")  # d3_r4
        return sorted(set(base + _hankel_cols(int(dtag[1:]), int(rtag[1:]), names)))
    return columns_for_operator_families(("operator",), names)


def sweep_segment(rebuild=False):
    rows = []
    best = {"score": -1.0, "k": 8, "kedmd": (1.0, 8, "s1", "r8"), "hankel": (3, 8)}
    for k in (8, 9):
        Xop, op_names, Xph, ph_names, y, groups, n_fish, meta = extract_segments(k, rebuild=rebuild)
        v4 = columns_for_operator_families(("operator_v4",), op_names)
        print(f"\n=== sweep k={k} v4 ===", flush=True)
        r = _stable5(Xop, y, groups, v4, [op_names[i] for i in v4])
        rows.append({"k": k, "cfg": "operator_v4", "stable5": r.get("macro_f1"), "nfeat": len(v4)})
        # structural add-ons
        for abl, fams in STRUCT_ABLS.items():
            cols = columns_for_operator_families(fams, op_names)
            print(f"  {abl} k={k}", flush=True)
            r = _stable5(Xop, y, groups, cols, [op_names[i] for i in cols])
            rows.append({"k": k, "cfg": abl, "stable5": r.get("macro_f1"), "nfeat": len(cols), "transient": None})
        # kedmd grid on top of v4+connection+hodge2
        base = columns_for_operator_families(("operator_v4", "connection", "hodge2"), op_names)
        for sm, rk, st, rt in KEDMD_CFGS:
            cols = sorted(set(base + _kedmd_cols(st, rt, op_names)))
            r = _stable5(Xop, y, groups, cols, [op_names[i] for i in cols])
            rec = {"k": k, "cfg": f"base+kedmd_{st}_{rt}", "stable5": r.get("macro_f1"), "nfeat": len(cols), "kedmd": (sm, rk)}
            rows.append(rec)
            print(f"  kedmd {st} {rt} k={k} F1={rec['stable5']}", flush=True)
        for delays, rk in HANKEL_CFGS:
            cols = sorted(set(base + _hankel_cols(delays, rk, op_names)))
            r = _stable5(Xop, y, groups, cols, [op_names[i] for i in cols])
            rec = {"k": k, "cfg": f"base+hdmd_d{delays}_r{rk}", "stable5": r.get("macro_f1"), "nfeat": len(cols), "hankel": (delays, rk)}
            rows.append(rec)
            print(f"  hankel d={delays} r={rk} k={k} F1={rec['stable5']}", flush=True)
        # full opt
        cols = columns_for_operator_families(("operator",), op_names)
        r = _stable5(Xop, y, groups, cols, [op_names[i] for i in cols], permute=0)
        rows.append({"k": k, "cfg": "operator_opt_all", "stable5": r.get("macro_f1"), "nfeat": len(cols)})
        print(f"  ALL k={k} F1={r.get('macro_f1')}", flush=True)

    def _sc(rec):
        v = rec.get("stable5")
        return -1.0 if v is None or not np.isfinite(v) else float(v)

    # freeze: best among base+kedmd / base+hankel / all / v4+families, preferring operator_opt_all if within 0.005
    candidates = [r for r in rows if r["cfg"] in {"operator_opt_all", "operator_v4"} or r["cfg"].startswith("base+")]
    winner = max(candidates, key=_sc)
    hyps = {
        "k": int(winner["k"]),
        "selection": winner["cfg"],
        "stable5": winner.get("stable5"),
        "kedmd_sigma_mult": 1.0,
        "kedmd_rank": 8,
        "hankel_delays": 3,
        "hankel_rank": 8,
        "seed": SEED,
        "note": "argmax segment grouped LOVO stable5 among v4 / structural add-ons / kedmd+hankel grids / all",
        "sweep_rows": rows,
    }
    if winner.get("kedmd"):
        hyps["kedmd_sigma_mult"], hyps["kedmd_rank"] = winner["kedmd"]
    if winner.get("hankel"):
        hyps["hankel_delays"], hyps["hankel_rank"] = winner["hankel"]
    # if ALL at same k is close, prefer ALL (uses every family)
    alls = [r for r in rows if r["cfg"] == "operator_opt_all" and r["k"] == winner["k"]]
    if alls and _sc(alls[0]) >= _sc(winner) - 0.005:
        hyps["k"] = int(alls[0]["k"])
        hyps["selection"] = "operator_opt_all"
        hyps["stable5"] = alls[0]["stable5"]
        hyps["use_all_families"] = True
    else:
        hyps["use_all_families"] = hyps["selection"] == "operator_opt_all"
    OUT.mkdir(parents=True, exist_ok=True)
    def _jsonable(x):
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]
        if isinstance(x, (np.floating,)):
            v = float(x)
            return v if np.isfinite(v) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, float):
            return x if np.isfinite(x) else None
        return x

    (OUT / "operator_opt_sweep.json").write_text(json.dumps(_jsonable(hyps), indent=2) + "\n")
    (OUT / "operator_opt_hyps.json").write_text(
        json.dumps(_jsonable({k: hyps[k] for k in hyps if k != "sweep_rows"}), indent=2) + "\n"
    )
    import csv as _csv

    with (OUT / "operator_opt_sweep.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["k", "cfg", "stable5", "nfeat", "kedmd", "hankel"])
        w.writeheader()
        for rec in rows:
            w.writerow(
                {
                    "k": rec.get("k"),
                    "cfg": rec.get("cfg"),
                    "stable5": rec.get("stable5"),
                    "nfeat": rec.get("nfeat"),
                    "kedmd": rec.get("kedmd") or "",
                    "hankel": rec.get("hankel") or "",
                }
            )
    print("froze", {k: hyps[k] for k in hyps if k != "sweep_rows"}, flush=True)
    return hyps


def _eval_block(Xop, y, groups, op_names, Xph, ph_names, *, permute=False, models=("xgb",), shap=False, hyps=None):
    out = {"n": int(len(y)), "n_videos": int(len(set(groups.tolist()))), "ablations": {}, "classifiers": {}}
    hyps = hyps or {"use_all_families": True, "selection": "operator_opt_all"}

    def run_all(X, cols, fn, model="xgb", perm=0, do_shap=False):
        block = {"n": int(len(y)), "tasks": {}}
        st = np.array([not is_transition(yi) for yi in y])
        jobs = {
            "stable5": (X[st][:, cols], y[st], groups[st]),
            "coarse6": (X[:, cols], collapse_transients(y), groups),
            "binary_transient": (X[:, cols], binary_transient(y), groups),
            "finegrained": (X[:, cols], y, groups),
        }
        for task, (Xt, yt, gt) in jobs.items():
            if task == "stable5" and (len(yt) < 16 or len(set(gt.tolist())) < 2):
                continue
            if task == "finegrained" and model != "xgb":
                continue
            p = perm if task == "stable5" else 0
            sh = do_shap if task == "stable5" else False
            rep = grouped_cv(Xt, yt, gt, model=model, feature_names=fn, permute_repeats=p, shap=sh)
            rep["n_features"] = len(fn)
            block["tasks"][task] = _slim(rep)
            extra = ""
            if task == "coarse6":
                extra = f" trans={rep.get('per_class_f1', {}).get('transient', float('nan')):.3f}"
            print(f"  {task:16s} {model:5s} nf={len(fn):3d} macroF1={rep.get('macro_f1', float('nan')):.3f}{extra}", flush=True)
        return block

    for abl, fams in STRUCT_ABLS.items():
        if abl == "operator_opt":
            cols = frozen_cols(hyps, op_names)
        else:
            cols = columns_for_operator_families(fams, op_names)
        fn = [op_names[i] for i in cols]
        print(f" ablation {abl} ({len(fn)})", flush=True)
        perm = 3 if permute and abl in {"operator_opt", "operator_v4"} else 0
        do_shap = shap and abl == "operator_opt"
        out["ablations"][abl] = run_all(Xop, cols, fn, perm=perm, do_shap=do_shap)
    cols_all = columns_for_operator_families(("operator",), op_names)
    fn_all = [op_names[i] for i in cols_all]
    print(f" ablation operator_all ({len(fn_all)})", flush=True)
    out["ablations"]["operator_all"] = run_all(Xop, cols_all, fn_all)
    for abl, fams in PHYS_ABLS.items():
        cols = columns_for_families(fams, ph_names)
        fn = [ph_names[i] for i in cols]
        print(f" ablation {abl} ({len(fn)})", flush=True)
        out["ablations"][abl] = run_all(Xph, cols, fn)
    cols_o = frozen_cols(hyps, op_names)
    cols_p = columns_for_families(("baseline", "fluct", "rate", "shape", "tunstrom", "fields", "early"), ph_names)
    Xc = np.hstack([Xop[:, cols_o], Xph[:, cols_p]])
    fn = [op_names[i] for i in cols_o] + [ph_names[i] for i in cols_p]
    print(f" ablation operator+physical ({len(fn)})", flush=True)
    out["ablations"]["operator+physical"] = run_all(Xc, list(range(len(fn))), fn)
    cols = frozen_cols(hyps, op_names)
    fn = [op_names[i] for i in cols]
    for model in models:
        print(f" classifier {model}", flush=True)
        out["classifiers"][model] = run_all(Xop, cols, fn, model=model)
    return out


def _events(X, names, y, groups, meta, min_dwell=3):
    use = [
        "hodge_rad_mean",
        "hodge_div_mean",
        "dyn_dlog_ell",
        "conn_frustr",
        "h2_curl_signed",
        "kedmd_vamp2",
        "koop_resid_mean",
        "spat_knn_ell_mean",
    ]
    cols = [names.index(n) for n in use if n in names]
    segs = load_real_segments()
    by = defaultdict(list)
    for s in segs:
        by[s["dataset"]].append(s)
    from src.classify.grouped import _inv_weights, _xgb
    from sklearn.preprocessing import LabelEncoder

    hmm_rows, dwell_rows, iid_rows = [], [], []
    for held in sorted(set(groups.tolist()), key=str):
        tr, te = groups != held, groups == held
        med = np.nanmedian(X[tr][:, cols], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        def fill(M):
            M = M.copy()
            bad = ~np.isfinite(M)
            if bad.any():
                M[bad] = np.take(med, np.where(bad)[1])
            return M

        Xs, ys = [], []
        for vid in sorted(set(groups[tr].tolist()), key=str):
            m = groups == vid
            if m.sum() < 8:
                continue
            order = np.argsort([int(meta[i][3]) for i in np.flatnonzero(m)])
            ix = np.flatnonzero(m)[order]
            lab = np.array([str(y[i]) if y[i] in CANONICAL else "" for i in ix], dtype=object)
            Xs.append(fill(X[ix][:, cols]))
            ys.append(lab)
        if len(Xs) < 2:
            continue
        hmm = SupervisedHMM()
        hmm.fit(Xs, ys)
        order = np.argsort([int(meta[i][3]) for i in np.flatnonzero(te)])
        ix = np.flatnonzero(te)[order]
        Xte = fill(X[ix][:, cols])
        t_mid = np.array([(int(meta[i][3]) + int(meta[i][4])) // 2 for i in ix])
        true_e = true_events(by[held], 10**9)
        for states, bucket in (
            (hmm.viterbi(Xte), hmm_rows),
            (hmm.viterbi_min_dwell(Xte, min_dwell=min_dwell), dwell_rows),
        ):
            pred_e = decoded_events(states, min_run=min_dwell)
            for e in pred_e:
                e["t"] = int(t_mid[min(int(e["t"]), len(t_mid) - 1)])
            bucket.append(match_events(true_e, pred_e, TOL))
        sttr = np.array([yi in CANONICAL for yi in y[tr]])
        if sttr.sum() < 8:
            continue
        le = LabelEncoder()
        le.fit(list(CANONICAL))
        clf = _xgb(len(CANONICAL))
        yenc = le.transform(y[tr][sttr])
        clf.fit(fill(X[tr][:, cols][sttr]), yenc, sample_weight=_inv_weights(yenc))
        pred = le.inverse_transform(np.clip(clf.predict(Xte), 0, 4))
        iid_e = decoded_events(pred, min_run=min_dwell)
        for e in iid_e:
            e["t"] = int(t_mid[min(int(e["t"]), len(t_mid) - 1)])
        iid_rows.append(match_events(true_e, iid_e, TOL))

    def _sum(rows):
        if not rows:
            return {}
        tp, fp, fn = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": float(prec), "recall": float(rec), "f1": float(f1), "n_true": sum(r["n_true"] for r in rows), "n_pred": sum(r["n_pred"] for r in rows)}

    return {"hmm": _sum(hmm_rows), "hmm_min_dwell": _sum(dwell_rows), "iid_windows": _sum(iid_rows), "tol_frames": TOL, "min_dwell": min_dwell}


def _f1(block, abl, task, key="macro_f1"):
    rec = block.get("ablations", {}).get(abl, {}).get("tasks", {}).get(task, {})
    if key == "transient_f1":
        return rec.get("per_class_f1", {}).get("transient", np.nan)
    return rec.get(key, np.nan)


def _plot_scales(combined):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    abls = ["phi4", "physical", "operator_v4", "connection", "hodge2", "kedmd", "operator_opt", "operator+physical"]
    scales = [s for s in ("segment", "w15", "w5") if s in combined]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), sharey=True)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(abls)))
    for ax, sc in zip(axes, scales):
        vals = [_f1(combined[sc]["pooled"], a, "stable5") for a in abls]
        ax.bar(np.arange(len(abls)), vals, color=cmap)
        ax.set_xticks(np.arange(len(abls)))
        ax.set_xticklabels(abls, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(sc)
        ax.axhline(0.8, color="0.6", ls="--", lw=0.8)
    axes[0].set_ylabel("pooled stable5 macro-F1")
    fig.suptitle("Optimized operators vs 4 Φ / physical / v4 baseline")
    fig.tight_layout()
    fig.savefig(FIG / "operator_opt_stable5_three_scales.png", dpi=150)
    plt.close(fig)


def run_scale(scale, k, rebuild=False, permute=False, hyps=None):
    if scale == "segment":
        Xop, op_names, Xph, ph_names, y, groups, n_fish, meta = extract_segments(k, rebuild=rebuild)
        tag = "segment"
    else:
        w, h = (15, 5) if scale == "w15" else (5, 5)
        Xop, op_names, y, groups, n_fish, meta = extract_windows(w, h, k, rebuild=rebuild)
        pz = np.load(PHYS_WIN(w, h), allow_pickle=True)
        Xph, ph_names = pz["X"], [str(n) for n in pz["names"]]
        tag = f"w{w}_h{h}"
    gl = _group_label(n_fish)
    summary = {
        "scale": scale,
        "k": k,
        "n_rows": int(len(y)),
        "n_videos": int(len(set(groups.tolist()))),
        "n_operator_features": len(op_names),
        "note": "Optimized operators. Unscaled kNN. No Φ. 150-fish pooled only.",
        "hyps": {k: v for k, v in (hyps or {}).items() if k != "sweep_rows"},
    }
    print(f"\n=== pooled opt {tag} n={len(y)} ===", flush=True)
    models = ("xgb", "logreg", "hgb") if scale == "segment" else ("xgb", "hgb")
    summary["pooled"] = _eval_block(
        Xop,
        y,
        groups,
        op_names,
        Xph,
        ph_names,
        permute=permute,
        models=models,
        shap=permute and scale == "segment",
        hyps=hyps,
    )
    summary["by_group"] = {}
    for g in sorted(set(gl.tolist()), key=str):
        m = gl == g
        print(f"\n=== {tag} {g} n={int(m.sum())} ===", flush=True)
        summary["by_group"][g] = _eval_block(
            Xop[m], y[m], groups[m], op_names, Xph[m], ph_names, permute=False, models=("xgb",), hyps=hyps
        )
    if scale != "segment":
        print("events…", flush=True)
        summary["events"] = _events(Xop, op_names, y, groups, meta, min_dwell=3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"operator_opt_lovo_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT / f"operator_opt_lovo_{tag}.json", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sweep", "lovo", "all"], default="all")
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    hyps_path = OUT / "operator_opt_hyps.json"
    if args.mode in {"sweep", "all"}:
        hyps = sweep_segment(rebuild=args.rebuild)
    else:
        hyps = json.loads(hyps_path.read_text()) if hyps_path.exists() else {"k": 8, "use_all_families": True}
    k = int(hyps.get("k", 8))
    operator_mod.K_NN = k
    if args.mode == "sweep":
        return
    summaries = {}
    for sc in ("segment", "w15", "w5"):
        summaries[sc] = run_scale(sc, k, rebuild=args.rebuild, permute=(sc == "segment"), hyps=hyps)
    (OUT / "operator_opt_three_scales.json").write_text(
        json.dumps({s: {"n_rows": v["n_rows"], "pooled_stable5_opt": _f1(v["pooled"], "operator_opt", "stable5"), "pooled_stable5_v4": _f1(v["pooled"], "operator_v4", "stable5"), "pooled_phi4": _f1(v["pooled"], "phi4", "stable5"), "pooled_physical": _f1(v["pooled"], "physical", "stable5")} for s, v in summaries.items()}, indent=2)
        + "\n"
    )
    _plot_scales(summaries)


if __name__ == "__main__":
    main()
