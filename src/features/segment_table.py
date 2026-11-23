"""Join official annotation labels to annotation_review kinematics.

Does not modify annotation files. Features were precomputed from the CouzinLab
trajectories during an earlier review pass; this only *reads* those tables and
aggregates them onto the official (dataset, start, end, label) intervals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.dataset import ROOT, load_official_segments

REVIEW_CSV = ROOT / "annotation_review" / "segment_review.csv"

KINEMATIC_COLS = (
    "phi_trans",
    "phi_tan",
    "phi_rad_pm",
    "phi_local",
    "P",
    "M",
    "dlogR",
    "rate",
    "vrad_norm",
    "R",
    "nnd",
    "speed",
    "dlogHull",
    "dlogNND",
    "frac_pol",
    "frac_mill",
    "frac_swarm",
)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def official_with_kinematics(path: Path | None = None) -> pd.DataFrame:
    segs = load_official_segments()
    off = pd.DataFrame(segs)
    sr = pd.read_csv(path or REVIEW_CSV)
    sr["dataset"] = sr["dataset"].astype(str).str.zfill(4)
    off["dataset"] = off["dataset"].astype(str).str.zfill(4)
    feat_cols = [c for c in KINEMATIC_COLS if c in sr.columns]
    rows = []
    for _, o in off.iterrows():
        sub = sr[sr["dataset"] == o["dataset"]]
        if sub.empty:
            continue
        w = np.array(
            [_overlap(int(o["start"]), int(o["end"]), int(r.start), int(r.end)) for r in sub.itertuples()],
            dtype=np.float64,
        )
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        rec = {
            "dataset": o["dataset"],
            "group": o["group"],
            "n_fish": int(str(o["group"]).split("_")[0]),
            "label": o["label"],
            "start": int(o["start"]),
            "end": int(o["end"]),
            "n_frames": int(o["end"]) - int(o["start"]),
            "has_trajectory": bool(o["has_trajectory"]),
            "n_review_overlap": int((w > 0).sum()),
        }
        for c in feat_cols:
            vals = sub[c].to_numpy(dtype=np.float64)
            rec[c] = float(np.nansum(w * vals))
        rows.append(rec)
    return pd.DataFrame(rows)
