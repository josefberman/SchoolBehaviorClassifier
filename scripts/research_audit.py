#!/usr/bin/env python3
"""Inventory annotations and evaluation pitfalls. Does not modify annotation files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.dataset import load_real_segments, load_positions_velocities
from src.features.windows import HOP_FRAMES, WINDOW_FRAMES, iter_window_bounds
from src.labels import CANONICAL, is_transition, ordered_labels
from src.sim.io import load_motion_json, mmss_to_frame

OUT = ROOT / "research" / "results"


def _segments_from_json() -> list[dict]:
    aliases_ok = []
    meta = json.loads((ROOT / "annotations" / "datasets.json").read_text())
    from src.labels import canonicalize, load_aliases

    aliases = load_aliases()
    for path in sorted((ROOT / "annotations").glob("*_motion.json")):
        if path.name.startswith("_"):
            continue
        ann = load_motion_json(path)
        ds = ann["dataset"]
        fps = float(ann.get("fps", meta[ds]["fps"]))
        group = meta[ds]["fish_group"]
        for i, seg in enumerate(ann["segments"]):
            raw = seg["label"]
            try:
                lab = canonicalize(raw, aliases)
            except ValueError:
                lab = f"UNKNOWN:{raw}"
            a = mmss_to_frame(seg["start"], fps)
            b = mmss_to_frame(seg["end"], fps)
            aliases_ok.append(
                {
                    "dataset": ds,
                    "group": group,
                    "seg": i,
                    "raw": raw,
                    "label": lab,
                    "start": a,
                    "end": b,
                    "n_frames": max(0, b - a),
                    "duration_s": max(0, b - a) / fps,
                    "fps": fps,
                    "is_transition": is_transition(lab) if not lab.startswith("UNKNOWN") else None,
                }
            )
    return aliases_ok


def audit() -> dict:
    segs = _segments_from_json()
    labels = Counter(s["label"] for s in segs)
    by_ds = defaultdict(Counter)
    by_group = defaultdict(Counter)
    frames = Counter()
    for s in segs:
        by_ds[s["dataset"]][s["label"]] += 1
        by_group[s["group"]][s["label"]] += 1
        frames[s["label"]] += s["n_frames"]

    unknown = [s for s in segs if str(s["label"]).startswith("UNKNOWN")]
    trans = [s for s in segs if s["is_transition"]]
    stable = [s for s in segs if s["is_transition"] is False]

    # Overlaps / gaps inside a video (possible annotation inconsistency).
    boundary_notes = []
    by_video = defaultdict(list)
    for s in segs:
        by_video[s["dataset"]].append(s)
    for ds, items in by_video.items():
        items = sorted(items, key=lambda z: (z["start"], z["end"]))
        for a, b in zip(items, items[1:]):
            if b["start"] < a["end"]:
                boundary_notes.append(
                    {
                        "dataset": ds,
                        "type": "overlap",
                        "a": (a["label"], a["start"], a["end"]),
                        "b": (b["label"], b["start"], b["end"]),
                        "overlap_frames": int(a["end"] - b["start"]),
                    }
                )
            elif b["start"] > a["end"]:
                gap = b["start"] - a["end"]
                if gap >= 2:
                    boundary_notes.append(
                        {
                            "dataset": ds,
                            "type": "gap",
                            "after": (a["label"], a["end"]),
                            "before": (b["label"], b["start"]),
                            "gap_frames": int(gap),
                            "gap_s": gap / a["fps"],
                        }
                    )

    loaded = load_real_segments()
    traj_present = sorted({s["dataset"] for s in loaded})
    window_counts = Counter()
    window_by_ds = defaultdict(Counter)
    for s in loaded:
        n = s["end"] - s["start"]
        n_w = len(iter_window_bounds(n, WINDOW_FRAMES, HOP_FRAMES))
        window_counts[s["label"]] += n_w
        window_by_ds[s["dataset"]][s["label"]] += n_w

    # Leakage notes for the published protocol.
    pitfalls = [
        "Published train/eval is sim-train → real-test (domain shift), not a real-data CV split.",
        "Windows are non-overlapping 5 frames (~0.17 s). Adjacent windows from the same segment "
        "are strongly dependent; a random split would leak. Grouped CV by video is required.",
        "The original GridSearchCV on sim uses a predefined train/test seed split (not grouped by N).",
        "windows.py already computes Φ std per window but AGG_FEATURE_NAMES drops it.",
        "YAML radii are 20/100/300 while full_run.sh locks 50/100/150 and DEFAULTS are 50/100/150.",
        "Sim N_VALUES include 50/100/200; real group sizes are 10/30/70/150.",
        "Sim fps=30 vs real 29.97; omega_max in YAML (~0.04–0.10) is far below annotated_config "
        "real-window omega_max (often 0.3–0.7).",
        "README says compaction has no real labels; current annotations do include compaction.",
        "Josef's SpectralSchooling fingerprint XGB reports macro-F1≈0.85 on a random 80/20 frame "
        "split (same videos in train and test) — that number is not leakage-safe.",
    ]

    report = {
        "n_segments": len(segs),
        "n_videos": len(by_video),
        "videos": sorted(by_video),
        "label_segment_counts": dict(labels),
        "label_frame_counts": dict(frames),
        "stable_segment_counts": dict(Counter(s["label"] for s in stable)),
        "transient_segment_counts": dict(Counter(s["label"] for s in trans)),
        "n_stable_segments": len(stable),
        "n_transient_segments": len(trans),
        "canonical_stable": list(CANONICAL),
        "transient_definition": "canonicalized labels containing '_to_'",
        "unknown_labels": unknown,
        "by_dataset_segments": {k: dict(v) for k, v in by_ds.items()},
        "by_group_segments": {k: dict(v) for k, v in by_group.items()},
        "boundary_issues_n": len(boundary_notes),
        "boundary_overlaps": [b for b in boundary_notes if b["type"] == "overlap"][:40],
        "n_gaps": sum(1 for b in boundary_notes if b["type"] == "gap"),
        "median_gap_s": float(
            np.median([b["gap_s"] for b in boundary_notes if b["type"] == "gap"])
        )
        if any(b["type"] == "gap" for b in boundary_notes)
        else None,
        "trajectories_available": traj_present,
        "n_loadable_segments": len(loaded),
        "window_counts_5frame": dict(window_counts),
        "window_by_dataset": {k: dict(v) for k, v in window_by_ds.items()},
        "window_frames": WINDOW_FRAMES,
        "hop_frames": HOP_FRAMES,
        "pitfalls": pitfalls,
        "published_real_eval": json.loads((ROOT / "results" / "real_eval_metrics.json").read_text())
        if (ROOT / "results" / "real_eval_metrics.json").exists()
        else None,
        "published_sim_eval": json.loads((ROOT / "results" / "sim_test_metrics.json").read_text())
        if (ROOT / "results" / "sim_test_metrics.json").exists()
        else None,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = audit()
    print(json.dumps({k: report[k] for k in report if k not in {"published_real_eval", "published_sim_eval", "boundary_overlaps", "by_dataset_segments"}}, indent=2))
    print(f"wrote {OUT / 'audit.json'}")


if __name__ == "__main__":
    main()
