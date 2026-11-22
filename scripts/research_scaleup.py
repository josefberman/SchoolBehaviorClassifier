#!/usr/bin/env python3
"""Leave-one-video-out on ALL official segments (10/30/70/150 fish).

Window-level 5-frame features still require on-disk H5. This script uses
overlap-weighted kinematics from annotation_review/segment_review.csv joined
onto official labels (annotations are not edited).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.grouped import binary_transient, collapse_transients, grouped_cv
from src.features.dataset import load_real_segments
from src.features.segment_table import KINEMATIC_COLS, official_with_kinematics
from src.labels import is_transition

OUT = ROOT / "research" / "results"

BASELINE = ("phi_trans", "phi_tan", "phi_rad_pm", "phi_local")
RATE = BASELINE + ("dlogR", "rate", "vrad_norm")
PHYSICAL = RATE + ("M", "R", "nnd", "speed", "P")
NONPHYS_PROXY = ("frac_pol", "frac_mill", "frac_swarm")  # composition, not graph spectra
ALL = tuple(c for c in KINEMATIC_COLS)

ABLATIONS = {
    "baseline_4phi": BASELINE,
    "baseline+dlogR": BASELINE + ("dlogR",),
    "baseline+rate": RATE,
    "physical": PHYSICAL,
    "all_kinematic": ALL,
}


def _xy(df, cols):
    use = [c for c in cols if c in df.columns]
    X = df[use].to_numpy(dtype=np.float64)
    y = df["label"].astype(str).to_numpy()
    g = df["dataset"].astype(str).to_numpy()
    return X, y, g, use


def _run_block(df, tag: str) -> dict:
    block = {
        "n": int(len(df)),
        "n_videos": int(df["dataset"].nunique()),
        "videos": sorted(df["dataset"].astype(str).unique()),
        "n_fish": sorted(set(int(x) for x in df["n_fish"])),
        "label_counts": df["label"].value_counts().to_dict(),
        "stable_counts": df.loc[~df["label"].map(is_transition), "label"].value_counts().to_dict(),
        "tasks": {},
    }
    if df["dataset"].nunique() < 2 or len(df) < 8:
        block["skipped"] = "too few videos or rows"
        return block
    stable = df[~df["label"].map(is_transition)].copy()
    for abl, cols in ABLATIONS.items():
        permute = 4 if abl in {"baseline_4phi", "baseline+dlogR", "physical"} else 0
        # stable5
        if len(stable) >= 8 and stable["dataset"].nunique() >= 2:
            X, y, g, use = _xy(stable, cols)
            rep = grouped_cv(X, y, g, model="xgb", feature_names=use, permute_repeats=permute)
            block["tasks"].setdefault("stable5", {})[abl] = {
                k: rep[k] for k in rep if k != "folds"
            }
            block["tasks"]["stable5"][abl]["folds"] = rep.get("folds")
        # coarse6
        X, y, g, use = _xy(df, cols)
        y6 = collapse_transients(y)
        rep = grouped_cv(X, y6, g, model="xgb", feature_names=use, permute_repeats=0)
        block["tasks"].setdefault("coarse6", {})[abl] = {k: rep[k] for k in rep if k != "folds"}
        # binary
        yb = binary_transient(y)
        rep = grouped_cv(X, yb, g, model="xgb", feature_names=use, permute_repeats=0)
        block["tasks"].setdefault("binary_transient", {})[abl] = {
            k: rep[k] for k in rep if k != "folds"
        }
        print(
            f"{tag:12s} {abl:18s} stable={block.get('tasks',{}).get('stable5',{}).get(abl,{}).get('macro_f1', float('nan')):.3f}",
            flush=True,
        )
    return block


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = official_with_kinematics()
    on_disk = load_real_segments()
    disk_ds = sorted({s["dataset"] for s in on_disk})
    report = {
        "n_official_joined": int(len(df)),
        "n_videos": int(df["dataset"].nunique()),
        "videos": sorted(df["dataset"].unique().tolist()),
        "by_group_n": df.groupby("group").size().to_dict(),
        "trajectories_on_disk": disk_ds,
        "note": (
            "Segment-level LOVO on official labels. Kinematics come from "
            "annotation_review/segment_review.csv, overlap-weighted onto official "
            "intervals. 30/70/150-fish H5 files are not on this VM (Drive download "
            "cap 10MB; files are 21–80MB and not publicly fetchable)."
        ),
        "pooled": {},
        "by_group": {},
    }
    report["pooled"] = _run_block(df, "pooled")
    for grp, sub in df.groupby("group"):
        report["by_group"][str(grp)] = _run_block(sub, str(grp))
    # 10-fish window-level reminder from existing ablation files
    w15 = OUT / "feature_ablation_w15_h5.json"
    if w15.exists():
        prev = json.loads(w15.read_text())
        report["window_level_10fish_w15_stable_baseline"] = (
            prev.get("tasks", {}).get("stable5", {}).get("baseline", {}).get("macro_f1")
        )
        report["window_level_10fish_w15_stable_rate"] = (
            prev.get("tasks", {}).get("stable5", {}).get("baseline+rate", {}).get("macro_f1")
        )
    (OUT / "scaleup_lovo.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT / 'scaleup_lovo.json'}")


if __name__ == "__main__":
    main()
