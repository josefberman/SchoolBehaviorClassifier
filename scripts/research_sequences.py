#!/usr/bin/env python3
"""HMM / CUSUM sequence models vs i.i.d. windows on 10-fish videos."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.grouped import grouped_cv
from src.classify.sequences import (
    SEQ_FEATURES,
    SupervisedHMM,
    classify_pieces,
    cusum_changes,
    decoded_events,
    frame_label_track,
    match_events,
    stack_features,
    stable_window_f1,
    true_events,
    window_from_states,
)
from src.features.dataset import load_official_segments, load_positions_velocities
from src.features.extended import compute_extended_series, extract_windows, family_feature_names
from src.labels import CANONICAL, is_transition

OUT = ROOT / "research" / "results"
TOL = 45  # frames ~1.5 s at 29.97 fps


def _by_dataset(segs):
    d = defaultdict(list)
    for s in segs:
        d[s["dataset"]].append(s)
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    segs = [s for s in load_official_segments() if s.get("has_trajectory")]
    by = _by_dataset(segs)
    videos = sorted(by)
    print("videos", videos, flush=True)

    loaded = {}
    series = {}
    Xf = {}
    yf = {}
    for ds, slist in by.items():
        path = ROOT / slist[0]["csv"]
        pos, vel = load_positions_velocities(path)
        loaded[ds] = (pos, vel)
        ser = compute_extended_series(pos, vel)
        series[ds] = ser
        Xf[ds] = stack_features(ser)
        yf[ds] = frame_label_track(pos.shape[0], slist)
        print(ds, "T", pos.shape[0], "N", pos.shape[1], flush=True)

    # --- i.i.d. window baseline (5-frame, 4 Φ) on the same videos ---
    names_phi = family_feature_names("baseline")
    wx, wy, wg = [], [], []
    win_bounds = defaultdict(list)
    for ds, slist in by.items():
        pos, vel = loaded[ds]
        for s in slist:
            a, b = s["start"], min(s["end"], pos.shape[0])
            Xw, names, bounds = extract_windows(pos[a:b], vel[a:b], window_frames=5, hop_frames=5)
            cols = [names.index(n) for n in names_phi if n in names]
            for row, (w0, w1) in zip(Xw, bounds):
                wx.append(row[cols])
                wy.append(s["label"])
                wg.append(ds)
                win_bounds[ds].append((a + w0, a + w1, s["label"]))
    Xw = np.vstack(wx)
    yw = np.array(wy, dtype=object)
    gw = np.array(wg, dtype=object)
    iid = grouped_cv(Xw, yw, gw, model="xgb", feature_names=names_phi)

    # LOVO HMM / CUSUM
    hmm_events = []
    cus_events = []
    iid_events = []  # window-label changes as a naive event detector
    hmm_win = {"true": [], "pred": []}
    cus_win = {"true": [], "pred": []}

    for held in videos:
        train_ds = [d for d in videos if d != held]
        hmm = SupervisedHMM()
        hmm.fit([Xf[d] for d in train_ds], [yf[d] for d in train_ds])
        pred_hmm = hmm.viterbi(Xf[held])
        # CUSUM centroids from train labeled frames
        cents = {}
        for lab in CANONICAL:
            acc = []
            for d in train_ds:
                m = yf[d] == lab
                if m.any():
                    acc.append(Xf[d][m])
            if acc:
                cents[lab] = np.concatenate(acc, axis=0).mean(axis=0)
            else:
                cents[lab] = np.zeros(Xf[held].shape[1])
        cuts = cusum_changes(Xf[held], threshold=7.5, min_run=18)
        pred_cus = classify_pieces(Xf[held], cuts, cents)

        te = true_events(by[held], len(yf[held]))
        pe_h = decoded_events(pred_hmm, min_run=15)
        pe_c = decoded_events(pred_cus, min_run=15)
        hmm_events.append(match_events(te, pe_h, TOL))
        cus_events.append(match_events(te, pe_c, TOL))

        # naive: treat i.i.d. predicted window class changes as events
        # (use true window labels' neighboring stables as a ceiling later)
        # Reconstruct predicted window sequence from grouped_cv OOF is heavy;
        # instead use majority of true windows' *iid model* via a fold-specific XGB.
        from src.classify.grouped import _inv_weights, _xgb
        from sklearn.preprocessing import LabelEncoder

        tr = gw != held
        te_m = gw == held
        le = LabelEncoder()
        ytr = yw[tr]
        present = [c for c in CANONICAL if c in set(ytr) or any(is_transition(yy) for yy in ytr)]
        # train on all labels present in train
        labs_tr = sorted(set(ytr.tolist()), key=str)
        le.fit(labs_tr)
        clf = _xgb(len(le.classes_))
        yenc = le.transform(ytr)
        Xtr, Xte = Xw[tr].copy(), Xw[te_m].copy()
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for mat in (Xtr, Xte):
            inds = ~np.isfinite(mat)
            if inds.any():
                mat[inds] = np.take(med, np.where(inds)[1])
        clf.fit(Xtr, yenc, sample_weight=_inv_weights(yenc))
        pred_w = le.inverse_transform(clf.predict(Xte))
        # map windows back to a frame track
        nT = len(yf[held])
        track = np.array([""] * nT, dtype=object)
        k = 0
        for a, b, lab in win_bounds[held]:
            if k >= len(pred_w):
                break
            track[a:b] = pred_w[k]
            k += 1
        # fill empties by hold
        last = "traveling"
        for t in range(nT):
            if track[t] == "" or is_transition(str(track[t])):
                track[t] = last
            else:
                last = str(track[t])
        iid_events.append(match_events(te, decoded_events(track, min_run=15), TOL))

        # window-level from sequence decode vs official window labels
        yt, yp = window_from_states(
            pred_hmm,
            [lab for _, _, lab in win_bounds[held]],
            [(a, b) for a, b, _ in win_bounds[held]],
        )
        hmm_win["true"].extend(yt.tolist())
        hmm_win["pred"].extend(yp.tolist())
        yt, yp = window_from_states(
            pred_cus,
            [lab for _, _, lab in win_bounds[held]],
            [(a, b) for a, b, _ in win_bounds[held]],
        )
        cus_win["true"].extend(yt.tolist())
        cus_win["pred"].extend(yp.tolist())
        print(
            f"held {held}  HMM event F1={hmm_events[-1]['f1']:.3f}  "
            f"CUSUM F1={cus_events[-1]['f1']:.3f}  iid-win F1={iid_events[-1]['f1']:.3f}",
            flush=True,
        )

    def _mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    def _sum_match(rows):
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        id_ok = sum(r["n_identity_correct"] for r in rows)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "identity_accuracy": float(id_ok / max(tp, 1)),
            "n_true": sum(r["n_true"] for r in rows),
            "n_pred": sum(r["n_pred"] for r in rows),
            "tol_frames": TOL,
            "per_video": rows,
        }

    out = {
        "videos": videos,
        "tol_frames": TOL,
        "tol_seconds": TOL / 29.97,
        "iid_window_grouped": {
            "macro_f1": iid.get("macro_f1"),
            "balanced_accuracy": iid.get("balanced_accuracy"),
            "stable": iid.get("stable"),
            "transient": iid.get("transient"),
            "per_class_f1": iid.get("per_class_f1"),
            "n": iid.get("n"),
        },
        "hmm": {
            "events": _sum_match(hmm_events),
            "window_stable": stable_window_f1(np.array(hmm_win["true"]), np.array(hmm_win["pred"])),
        },
        "cusum": {
            "events": _sum_match(cus_events),
            "window_stable": stable_window_f1(np.array(cus_win["true"]), np.array(cus_win["pred"])),
        },
        "iid_window_as_events": _sum_match(iid_events),
    }
    (OUT / "sequences.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"HMM event F1={out['hmm']['events']['f1']:.3f}  "
        f"CUSUM={out['cusum']['events']['f1']:.3f}  "
        f"iid-events={out['iid_window_as_events']['f1']:.3f}  "
        f"HMM win-stable macro={out['hmm']['window_stable'].get('macro_f1', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
