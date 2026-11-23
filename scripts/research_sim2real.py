#!/usr/bin/env python3
"""Train on simulated 5-frame windows, test on real (sim-to-real transfer)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.grouped import _xgb, _inv_weights
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from src.features.extended import (
    columns_for_families,
    extract_windows,
    family_feature_names,
)
from src.labels import BEHAVIOR_SHORT, CANONICAL, ordered_labels
from src.sim.model_fast import run_simulation_fast
from src.sim.realistic import run_realistic

OUT = ROOT / "research" / "results"
CACHE = ROOT / "research" / "cache"


def _real_stable(window: int = 5, hop: int = 5):
    path = CACHE / f"largen_windows_w{window}_h{hop}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    cols = columns_for_families(("baseline",), names)
    y = z["y"]
    from src.labels import is_transition

    m = np.array([not is_transition(yi) and yi in CANONICAL for yi in y])
    n_fish = z["n_fish"]
    return z["X"][m][:, cols], y[m], n_fish[m], names


def _sim_xy(engine: str, n: int, seeds: int, cfg_dir: Path | None):
    names = family_feature_names("baseline")
    xs, ys = [], []
    for lab in CANONICAL:
        ov = None
        if cfg_dir is not None:
            p = cfg_dir / f"{BEHAVIOR_SHORT[lab]}.yaml"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    ov = yaml.safe_load(f) or {}
        for seed in range(seeds):
            if engine == "original":
                res = run_simulation_fast(lab, n, seed)
            else:
                res = run_realistic(
                    lab, n, seed, overrides=ov, prefer_fit=(engine == "fit")
                )
            Xw, feat_names, _ = extract_windows(
                res.positions, res.velocities, window_frames=5, hop_frames=5
            )
            cols = [feat_names.index(c) for c in names if c in feat_names]
            for row in Xw:
                xs.append(row[cols])
                ys.append(lab)
    return np.vstack(xs), np.array(ys, dtype=object)


def _eval(Xtr, ytr, Xte, yte) -> dict:
    labels = ordered_labels(set(ytr) | set(yte))
    le = LabelEncoder()
    le.fit(list(ytr))
    known = np.array([yi in le.classes_ for yi in yte])
    clf = _xgb(len(le.classes_))
    yenc = le.transform(ytr)
    clf.fit(Xtr, yenc, sample_weight=_inv_weights(yenc))
    pred = np.array(["__unseen__"] * len(yte), dtype=object)
    pred[known] = le.inverse_transform(np.clip(clf.predict(Xte[known]), 0, len(le.classes_) - 1))
    return {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "macro_f1": float(f1_score(yte, pred, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "per_class_f1": {
            lab: float(f1_score(yte, pred, average=None, labels=[lab], zero_division=0)[0])
            for lab in labels
        },
        "classification_report": classification_report(
            yte, pred, labels=labels, zero_division=0, output_dict=True
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seeds", type=int, default=8)
    args = p.parse_args()
    Xr, yr, nr, _ = _real_stable()
    report = {"real_n": int(len(yr)), "real_n_10fish": int(np.sum(nr <= 12)), "engines": {}}
    engines = [
        ("original", None),
        ("fit", ROOT / "configs" / "behaviors_fit"),
    ]
    long_dir = ROOT / "configs" / "behaviors_cma_long"
    if long_dir.exists() and any(long_dir.glob("*.yaml")):
        engines.append(("cma_long", long_dir))
    else:
        cma_dir = ROOT / "configs" / "behaviors_cma"
        if cma_dir.exists():
            engines.append(("cma", cma_dir))
    for name, cfg_dir in engines:
        print(f"sim-to-real {name}", flush=True)
        prefer = "fit" if name == "fit" else name
        Xs, ys = _sim_xy("original" if name == "original" else prefer, args.n, args.seeds, cfg_dir)
        rec = _eval(Xs, ys, Xr, yr)
        rec10 = _eval(Xs, ys, Xr[nr <= 12], yr[nr <= 12])
        rec["on_10fish"] = rec10
        report["engines"][name] = rec
        print(
            f"  all-N macroF1={rec['macro_f1']:.3f}  10-fish={rec10['macro_f1']:.3f}  "
            f"n_train={rec['n_train']}",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sim2real.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT / "sim2real.json")


if __name__ == "__main__":
    main()
