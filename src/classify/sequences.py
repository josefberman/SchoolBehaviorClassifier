"""Sequence models for transients: supervised HMM and CUSUM change-points."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from src.labels import CANONICAL, is_transition


SEQ_FEATURES = ("phi_trans", "phi_tan", "phi_rad_pm", "M", "dlogR")


def frame_label_track(n_frames: int, segments: list[dict]) -> np.ndarray:
    """Per-frame official labels; unlabeled frames are empty strings."""
    y = np.array([""] * n_frames, dtype=object)
    for s in segments:
        a, b = int(s["start"]), min(int(s["end"]), n_frames)
        y[a:b] = s["label"]
    return y


def _smooth(x: np.ndarray, win: int = 9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < win:
        return x
    k = np.ones(win) / win
    return np.convolve(np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0), k, mode="same")


def stack_features(series: dict[str, np.ndarray], names: tuple[str, ...] = SEQ_FEATURES) -> np.ndarray:
    cols = []
    for n in names:
        x = np.asarray(series[n], dtype=np.float64)
        x = _smooth(x, 9)
        med = np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, med)
        cols.append(x)
    X = np.stack(cols, axis=1)
    return X


def true_events(segments: list[dict], n_frames: int) -> list[dict]:
    """A→B events at the start of each official transient segment."""
    ev = []
    for s in segments:
        if not is_transition(s["label"]):
            continue
        a, b = s["label"].split("_to_", 1)
        t0, t1 = int(s["start"]), min(int(s["end"]), n_frames)
        ev.append({"src": a, "dst": b, "t": t0, "t_mid": (t0 + t1) // 2, "t_end": t1})
    return ev


def decoded_events(states: np.ndarray, min_run: int = 12) -> list[dict]:
    """State changes after collapsing short runs."""
    if len(states) == 0:
        return []
    runs = []
    i = 0
    while i < len(states):
        j = i + 1
        while j < len(states) and states[j] == states[i]:
            j += 1
        if (j - i) >= min_run or not runs:
            runs.append((i, j, str(states[i])))
        else:
            # absorb short flicker into previous run
            s0, s1, lab = runs[-1]
            runs[-1] = (s0, j, lab)
        i = j
    # merge identical neighbors
    merged = []
    for r in runs:
        if merged and merged[-1][2] == r[2]:
            merged[-1] = (merged[-1][0], r[1], r[2])
        else:
            merged.append(r)
    ev = []
    for (s0, e0, a), (s1, e1, b) in zip(merged, merged[1:]):
        if a != b and a in CANONICAL and b in CANONICAL:
            ev.append({"src": a, "dst": b, "t": s1, "t_mid": s1, "t_end": e1})
    return ev


def match_events(true_e: list[dict], pred_e: list[dict], tol: int) -> dict:
    used = set()
    tp = 0
    id_ok = 0
    for te in true_e:
        best = None
        best_d = None
        for j, pe in enumerate(pred_e):
            if j in used:
                continue
            d = abs(int(pe["t"]) - int(te["t"]))
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is None:
            continue
        used.add(best)
        tp += 1
        pe = pred_e[best]
        if pe["src"] == te["src"] and pe["dst"] == te["dst"]:
            id_ok += 1
    fp = len(pred_e) - len(used)
    fn = len(true_e) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "n_true": len(true_e),
        "n_pred": len(pred_e),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "identity_accuracy": float(id_ok / max(tp, 1)),
        "n_identity_correct": id_ok,
        "tol_frames": int(tol),
    }


class SupervisedHMM:
    """Diagonal-Gaussian HMM with 5 stable states; transitions from data + epsilon."""

    def __init__(self, states: tuple[str, ...] = CANONICAL, eps: float = 1e-3):
        self.states = list(states)
        self.eps = float(eps)
        self.mu = None
        self.var = None
        self.logA = None
        self.logpi = None

    def fit(self, Xs: list[np.ndarray], ys: list[np.ndarray]) -> "SupervisedHMM":
        d = Xs[0].shape[1]
        k = len(self.states)
        idx = {s: i for i, s in enumerate(self.states)}
        mu = np.zeros((k, d))
        var = np.ones((k, d))
        counts = np.zeros(k)
        trans = np.full((k, k), self.eps)
        pi = np.full(k, self.eps)
        for X, y in zip(Xs, ys):
            last = None
            started = False
            for t, lab in enumerate(y):
                if lab not in idx:
                    last = None
                    continue
                i = idx[lab]
                mu[i] += X[t]
                counts[i] += 1
                if not started:
                    pi[i] += 1.0
                    started = True
                if last is not None:
                    trans[last, i] += 1.0
                last = i
        for i in range(k):
            if counts[i] > 0:
                mu[i] /= counts[i]
            acc = np.zeros(d)
            for X, y in zip(Xs, ys):
                m = np.array([yi == self.states[i] for yi in y])
                if m.any():
                    acc += np.sum((X[m] - mu[i]) ** 2, axis=0)
            var[i] = np.maximum(acc / max(counts[i], 1.0), 1e-4)
        trans = trans / trans.sum(axis=1, keepdims=True)
        pi = pi / pi.sum()
        self.mu, self.var = mu, var
        self.logA = np.log(trans)
        self.logpi = np.log(pi)
        self.counts = counts
        return self

    def _logB(self, X: np.ndarray) -> np.ndarray:
        # (T, K)
        T, d = X.shape
        k = len(self.states)
        out = np.zeros((T, k))
        for i in range(k):
            diff = X - self.mu[i]
            out[:, i] = -0.5 * (
                np.sum(np.log(2 * np.pi * self.var[i])) + np.sum(diff**2 / self.var[i], axis=1)
            )
        return out

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        logB = self._logB(X)
        T, k = logB.shape
        delta = np.full((T, k), -np.inf)
        psi = np.zeros((T, k), dtype=np.int32)
        delta[0] = self.logpi + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + self.logA
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores.max(axis=0) + logB[t]
        path = np.zeros(T, dtype=np.int32)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return np.array([self.states[i] for i in path], dtype=object)

    def viterbi_min_dwell(self, X: np.ndarray, min_dwell: int = 3) -> np.ndarray:
        """Viterbi that forbids leaving a state before ``min_dwell`` steps."""
        D = max(int(min_dwell), 1)
        if D <= 1:
            return self.viterbi(X)
        logB = self._logB(X)
        T, k = logB.shape
        # expanded state (label, age) with age in 0..D-1; age D-1 means "can leave"
        n = k * D
        delta = np.full((T, n), -np.inf)
        psi = np.zeros((T, n), dtype=np.int32)

        def idx(lab, age):
            return lab * D + age

        for i in range(k):
            delta[0, idx(i, 0)] = self.logpi[i] + logB[0, i]
        for t in range(1, T):
            for j in range(k):
                # age 0: switch in from a different mature state
                enter = np.full(k, -np.inf)
                for i in range(k):
                    if i == j:
                        continue
                    enter[i] = delta[t - 1, idx(i, D - 1)] + self.logA[i, j]
                i_best = int(np.argmax(enter))
                delta[t, idx(j, 0)] = enter[i_best] + logB[t, j]
                psi[t, idx(j, 0)] = idx(i_best, D - 1)
                # ages 1 .. D-2: must continue the same label
                for age in range(1, D - 1):
                    src = idx(j, age - 1)
                    delta[t, idx(j, age)] = delta[t - 1, src] + self.logA[j, j] + logB[t, j]
                    psi[t, idx(j, age)] = src
                # mature: arrive from age D-2 or stay mature
                cand_src = [idx(j, D - 1)]
                cand_sc = [delta[t - 1, idx(j, D - 1)] + self.logA[j, j]]
                if D >= 2:
                    cand_src.append(idx(j, D - 2))
                    cand_sc.append(delta[t - 1, idx(j, D - 2)] + self.logA[j, j])
                b = int(np.argmax(cand_sc))
                delta[t, idx(j, D - 1)] = cand_sc[b] + logB[t, j]
                psi[t, idx(j, D - 1)] = cand_src[b]
        last = int(np.argmax(delta[-1]))
        path_e = np.zeros(T, dtype=np.int32)
        path_e[-1] = last
        for t in range(T - 2, -1, -1):
            path_e[t] = psi[t + 1, path_e[t + 1]]
        return np.array([self.states[i // D] for i in path_e], dtype=object)


def cusum_changes(X: np.ndarray, threshold: float = 8.0, min_run: int = 20) -> list[int]:
    """Unsupervised mean-shift CUSUM on standardized feature trajectories."""
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-6)
    T = len(Z)
    if T < min_run * 2:
        return []
    # Two-sided CUSUM on L2 increment.
    s_pos = 0.0
    s_neg = 0.0
    last = 0
    times = []
    drift = 0.15
    for t in range(1, T):
        d = float(np.linalg.norm(Z[t] - Z[t - 1]))
        s_pos = max(0.0, s_pos + d - drift)
        s_neg = min(0.0, s_neg - d + drift)
        if (s_pos > threshold or -s_neg > threshold) and (t - last) >= min_run:
            times.append(t)
            last = t
            s_pos = s_neg = 0.0
    return times


def classify_pieces(
    X: np.ndarray, cuts: list[int], centroids: dict[str, np.ndarray]
) -> np.ndarray:
    T = len(X)
    bounds = [0] + sorted(cuts) + [T]
    states = np.empty(T, dtype=object)
    names = list(centroids)
    C = np.stack([centroids[n] for n in names], axis=0)
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        mu = X[a:b].mean(axis=0)
        j = int(np.argmin(np.linalg.norm(C - mu, axis=1)))
        states[a:b] = names[j]
    return states


def window_from_states(states: np.ndarray, y_win: np.ndarray, bounds: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    pred = []
    true = []
    for (a, b), lab in zip(bounds, y_win):
        sl = states[a:b]
        if sl.size == 0:
            continue
        # majority
        vals, counts = np.unique(sl, return_counts=True)
        pred.append(vals[int(np.argmax(counts))])
        true.append(lab)
    return np.array(true, dtype=object), np.array(pred, dtype=object)


def stable_window_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.array([not is_transition(str(y)) and str(y) in CANONICAL for y in y_true])
    if not mask.any():
        return {}
    labels = list(CANONICAL)
    return {
        "n": int(mask.sum()),
        "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", labels=labels, zero_division=0)),
        "per_class_f1": {
            lab: float(f1_score(y_true[mask], y_pred[mask], average=None, labels=[lab], zero_division=0)[0])
            for lab in labels
        },
    }
