"""Extended per-frame and window features for leakage-safe classification.

Original classifier inputs are the 5-frame means of four order parameters
(``phi_trans``, ``phi_tan``, ``phi_rad_pm``, ``phi_local``). This module keeps
those as family ``baseline`` and adds:

* ``fluct`` — window std of the four Φ (already computed in windows.py, unused)
* ``rate`` — slopes + d log R / dt (transients / early-warning)
* ``shape`` — speed, NND, gyration radius, aspect ratio, density
* ``tunstrom`` — unsigned rotation M, local–global mismatch
* ``spectral`` — graph Laplacian λ2 / gap on kNN and metric graphs (non-physical)
* ``corr`` — connected velocity correlation length (Cavagna-style)
* ``info`` — heading-histogram entropy
* ``fields`` — COM-frame vorticity / divergence proxies, front–back density

All series are NaN-aware. Windowing here is independent of
``src.features.windows.WINDOW_FRAMES`` so 5- and 15-frame experiments can share
the same extractor without changing the original 5-frame-only path.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.features.order_params import K_LOCAL

BASELINE_SERIES = ("phi_trans", "phi_tan", "phi_rad_pm", "phi_local")

# Cheap per-frame series (vectorized). Spectral/corr are window-level.
SERIES_NAMES = (
    *BASELINE_SERIES,
    "M",
    "mean_speed",
    "speed_cv",
    "nnd",
    "R_gyr",
    "aspect",
    "density",
    "heading_entropy",
    "com_speed",
    "mismatch",
    "dlogR",
    "abs_vorticity",
    "divergence",
    "front_back",
)

FAMILY_SERIES: dict[str, tuple[str, ...]] = {
    "baseline": BASELINE_SERIES,
    "fluct": BASELINE_SERIES,  # uses _std of these
    "rate": BASELINE_SERIES + ("dlogR",),
    "shape": ("mean_speed", "speed_cv", "nnd", "R_gyr", "aspect", "density", "com_speed"),
    "tunstrom": ("M", "mismatch"),
    "info": ("heading_entropy",),
    "fields": ("abs_vorticity", "divergence", "front_back"),
}

SPECTRAL_WINDOW_NAMES = (
    "lambda2_knn",
    "gap_knn",
    "lambda2_metric",
    "gap_metric",
    "n_components_metric",
)
CORR_WINDOW_NAMES = ("corr_length", "susceptibility")
EARLY_WINDOW_NAMES = ("phi_trans_ac1",)


def family_feature_names(
    family: str,
    *,
    aggs: tuple[str, ...] = ("mean",),
) -> list[str]:
    """Column names for one ablation family (plus window-only families)."""
    if family == "baseline":
        return [f"{n}_mean" for n in BASELINE_SERIES]
    if family == "fluct":
        return [f"{n}_std" for n in BASELINE_SERIES]
    if family == "rate":
        names = [f"{n}_slope" for n in BASELINE_SERIES]
        names.append("dlogR_mean")
        return names
    if family == "spectral":
        return list(SPECTRAL_WINDOW_NAMES)
    if family == "corr":
        return list(CORR_WINDOW_NAMES)
    if family == "early":
        return list(EARLY_WINDOW_NAMES)
    series = FAMILY_SERIES[family]
    out: list[str] = []
    for n in series:
        for a in aggs:
            out.append(f"{n}_{a}")
    return out


def all_window_feature_names() -> list[str]:
    names: list[str] = []
    for series in SERIES_NAMES:
        names.extend([f"{series}_mean", f"{series}_std", f"{series}_slope"])
    names.extend(SPECTRAL_WINDOW_NAMES)
    names.extend(CORR_WINDOW_NAMES)
    names.extend(EARLY_WINDOW_NAMES)
    return names


def _unit_vectors(vel: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speeds = np.linalg.norm(vel, axis=-1)
    hat = np.zeros_like(vel)
    moving = valid & (speeds > 1e-9)
    hat[moving] = vel[moving] / speeds[moving, None]
    return hat, speeds


def _masked_mean(x: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    w = valid.astype(np.float64)
    num = np.nansum(np.where(valid, x, 0.0), axis=axis)
    den = np.maximum(w.sum(axis=axis), 1e-12)
    out = num / den
    out[w.sum(axis=axis) == 0] = np.nan
    return out


def compute_extended_series(positions: np.ndarray, velocities: np.ndarray) -> dict[str, np.ndarray]:
    """NaN-aware per-frame features. positions/velocities: (T, N, 2)."""
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[None, ...]
        vel = vel[None, ...]
    t, n, _ = pos.shape
    valid = np.isfinite(pos).all(axis=-1) & np.isfinite(vel).all(axis=-1)
    hat_v, speeds = _unit_vectors(vel, valid)
    n_valid = valid.sum(axis=1).astype(np.float64)

    # Polarization Φ_trans = ||⟨v̂⟩|| over valid fish.
    mean_hat = np.stack(
        [
            _masked_mean(hat_v[..., 0], valid, axis=1),
            _masked_mean(hat_v[..., 1], valid, axis=1),
        ],
        axis=-1,
    )
    phi_trans = np.linalg.norm(mean_hat, axis=-1)

    centroid = np.stack(
        [
            _masked_mean(pos[..., 0], valid, axis=1),
            _masked_mean(pos[..., 1], valid, axis=1),
        ],
        axis=-1,
    )
    q = pos - centroid[:, None, :]
    q_norm = np.linalg.norm(q, axis=-1)
    r_hat = np.zeros_like(q)
    safe = valid & (q_norm > 1e-9)
    r_hat[safe] = q[safe] / q_norm[safe, None]

    # Centered Φ_rad / Φ_tan (same construction as order_params.py, NaN-aware).
    v_p = hat_v - mean_hat[:, None, :]
    r_mean = np.stack(
        [
            _masked_mean(r_hat[..., 0], valid, axis=1),
            _masked_mean(r_hat[..., 1], valid, axis=1),
        ],
        axis=-1,
    )
    r_p = r_hat - r_mean[:, None, :]
    v_p = np.where(valid[..., None], v_p, 0.0)
    r_p = np.where(valid[..., None], r_p, 0.0)
    sum_v2 = np.sum(v_p[..., 0] ** 2 + v_p[..., 1] ** 2, axis=1)
    sum_r2 = np.sum(r_p[..., 0] ** 2 + r_p[..., 1] ** 2, axis=1)
    denom = np.sqrt(sum_v2 * sum_r2)
    phi_rad = np.full(t, np.nan)
    phi_tan = np.full(t, np.nan)
    good = denom > 1e-12
    phi_rad[good] = np.sum(v_p * r_p, axis=(1, 2))[good] / denom[good]
    cross_z = r_p[..., 0] * v_p[..., 1] - r_p[..., 1] * v_p[..., 0]
    phi_tan[good] = np.abs(np.sum(cross_z, axis=1)[good]) / denom[good]

    # Tunstrom rotation M = |⟨r̂ × v̂⟩|.
    cross_raw = r_hat[..., 0] * hat_v[..., 1] - r_hat[..., 1] * hat_v[..., 0]
    M = np.abs(_masked_mean(cross_raw, valid, axis=1))

    mean_speed = _masked_mean(speeds, valid, axis=1)
    speed_std = np.sqrt(
        np.maximum(
            _masked_mean((speeds - mean_speed[:, None]) ** 2, valid, axis=1),
            0.0,
        )
    )
    speed_cv = speed_std / np.maximum(mean_speed, 1e-9)

    R_gyr = np.sqrt(_masked_mean(q_norm**2, valid, axis=1))
    # Aspect ratio from valid-position covariance eigenvalues.
    aspect = np.full(t, np.nan)
    for i in range(t):
        vi = valid[i]
        if int(vi.sum()) < 3:
            continue
        cov = np.cov(pos[i, vi].T)
        evals = np.sort(np.linalg.eigvalsh(cov))
        aspect[i] = float(
            np.clip(np.sqrt(max(evals[-1], 0.0) / max(evals[0], 1e-6)), 1.0, 20.0)
        )

    density = n_valid / np.maximum(np.pi * R_gyr**2, 1.0)
    com_vel = np.stack(
        [
            _masked_mean(vel[..., 0], valid, axis=1),
            _masked_mean(vel[..., 1], valid, axis=1),
        ],
        axis=-1,
    )
    com_speed = np.linalg.norm(com_vel, axis=-1)

    # Heading entropy (8 bins).
    heading_entropy = np.full(t, np.nan)
    angles = np.arctan2(hat_v[..., 1], hat_v[..., 0])
    bins = np.linspace(-np.pi, np.pi, 9)
    for i in range(t):
        a = angles[i, valid[i] & (speeds[i] > 1e-9)]
        if a.size < 3:
            continue
        hist, _ = np.histogram(a, bins=bins)
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        heading_entropy[i] = float(-(p * np.log(p)).sum() / np.log(8.0))

    dlogR = np.full(t, np.nan)
    if t >= 2:
        dlogR[1:] = np.diff(np.log(np.maximum(R_gyr, 1e-6)))

    phi_local, nnd, abs_vort, diverg, front_back = _pairwise_frame_features(
        pos, hat_v, vel, valid, mean_hat
    )
    mismatch = phi_local - phi_trans

    return {
        "phi_trans": phi_trans,
        "phi_tan": phi_tan,
        "phi_rad_pm": phi_rad,
        "phi_local": phi_local,
        "M": M,
        "mean_speed": mean_speed,
        "speed_cv": speed_cv,
        "nnd": nnd,
        "R_gyr": R_gyr,
        "aspect": aspect,
        "density": density,
        "heading_entropy": heading_entropy,
        "com_speed": com_speed,
        "mismatch": mismatch,
        "dlogR": dlogR,
        "abs_vorticity": abs_vort,
        "divergence": diverg,
        "front_back": front_back,
    }


def _pairwise_frame_features(
    pos: np.ndarray,
    hat_v: np.ndarray,
    vel: np.ndarray,
    valid: np.ndarray,
    mean_hat: np.ndarray,
    k: int = K_LOCAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local polarization, NND, vorticity, divergence, front–back density."""
    t, n, _ = pos.shape
    phi_local = np.full(t, np.nan)
    nnd = np.full(t, np.nan)
    abs_vort = np.full(t, np.nan)
    diverg = np.full(t, np.nan)
    front_back = np.full(t, np.nan)
    idx = np.arange(n)
    step = 32 if n * n * 32 * 8 < 256 * 1024 * 1024 else max(1, 4_000_000 // max(n * n, 1))
    for start in range(0, t, step):
        sl = slice(start, min(t, start + step))
        block = pos[sl]
        hv = hat_v[sl]
        vv = vel[sl]
        val = valid[sl]
        tb = block.shape[0]
        delta = block[:, :, None, :] - block[:, None, :, :]
        d2 = np.einsum("tijn,tijn->tij", delta, delta)
        pair_ok = val[:, :, None] & val[:, None, :]
        d2 = np.where(pair_ok, d2, np.inf)
        d2[:, idx, idx] = np.inf
        dist = np.sqrt(d2)

        # NND: mean nearest-neighbor distance among valid fish.
        nn_d = np.min(dist, axis=-1)
        nnd[sl] = np.nanmean(np.where(val, nn_d, np.nan), axis=1)

        k_use = min(int(k), max(n - 1, 1))
        finite_counts = np.isfinite(d2).sum(axis=-1)
        k_frame = np.minimum(k_use, np.maximum(finite_counts, 0))
        nn = np.argpartition(d2, kth=min(k_use, n - 1) - 1 if n > 1 else 0, axis=-1)[..., :k_use]
        t_idx = np.arange(tb)[:, None, None]
        neigh_v = hv[t_idx, nn]
        # Mask neighbors that were inf.
        neigh_ok = np.take_along_axis(np.isfinite(d2), nn, axis=-1)
        neigh_v = np.where(neigh_ok[..., None], neigh_v, np.nan)
        local_pol = np.nanmean(neigh_v, axis=2)
        local_norm = np.linalg.norm(local_pol, axis=-1)
        phi_local[sl] = np.nanmean(np.where(val, local_norm, np.nan), axis=1)

        # Vorticity / divergence vs 5-NN in COM-subtracted velocity.
        rel_v = vv - np.nanmean(np.where(val[..., None], vv, np.nan), axis=1, keepdims=True)
        neigh_rel_v = rel_v[t_idx, nn]
        neigh_dxy = block[t_idx, nn] - block[:, :, None, :]
        cross = neigh_dxy[..., 0] * neigh_rel_v[..., 1] - neigh_dxy[..., 1] * neigh_rel_v[..., 0]
        r2 = np.maximum((neigh_dxy**2).sum(axis=-1), 1e-6)
        vort_i = np.nanmean(np.where(neigh_ok, cross / r2, np.nan), axis=-1)
        div_i = np.nanmean(
            np.where(
                neigh_ok,
                (neigh_dxy[..., 0] * neigh_rel_v[..., 0] + neigh_dxy[..., 1] * neigh_rel_v[..., 1])
                / r2,
                np.nan,
            ),
            axis=-1,
        )
        abs_vort[sl] = np.nanmean(np.where(val, np.abs(vort_i), np.nan), axis=1)
        diverg[sl] = np.nanmean(np.where(val, div_i, np.nan), axis=1)

        # Front–back: density along the group heading axis.
        axis = mean_hat[sl]
        proj = np.einsum("tnd,td->tn", block - np.nanmean(np.where(val[..., None], block, np.nan), axis=1, keepdims=True), axis)
        front = val & (proj > 0)
        back = val & (proj < 0)
        n_f = front.sum(axis=1).astype(np.float64)
        n_b = back.sum(axis=1).astype(np.float64)
        front_back[sl] = (n_f - n_b) / np.maximum(n_f + n_b, 1.0)
    return phi_local, nnd, abs_vort, diverg, front_back


def _graph_spectrum(pos: np.ndarray, valid: np.ndarray, *, k: int = 5, radius: float | None = None) -> tuple[float, float, float]:
    """λ2, spectral gap, connected-component count for one frame."""
    idx = np.flatnonzero(valid)
    m = int(idx.size)
    if m < 3:
        return np.nan, np.nan, np.nan
    xy = pos[idx]
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = np.zeros((m, m), dtype=np.float64)
    if radius is not None:
        adj[d <= radius] = 1.0
    else:
        kk = min(k, m - 1)
        nn = np.argpartition(d, kth=kk - 1, axis=-1)[:, :kk]
        rows = np.repeat(np.arange(m), kk)
        adj[rows, nn.ravel()] = 1.0
        adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(np.real(evals))
    lam2 = float(evals[1]) if evals.size > 1 else np.nan
    gap = float(evals[2] - evals[1]) if evals.size > 2 else np.nan
    n_comp = float(np.sum(evals < 1e-8))
    return lam2, gap, n_comp


def _corr_length(pos: np.ndarray, vel: np.ndarray, valid: np.ndarray, n_bins: int = 8) -> tuple[float, float]:
    """Distance where connected velocity correlation drops to 1/e, plus χ = N Var(Φ)."""
    idx = np.flatnonzero(valid)
    m = int(idx.size)
    if m < 4:
        return np.nan, np.nan
    xy = pos[idx]
    v = vel[idx]
    speed = np.linalg.norm(v, axis=-1)
    moving = speed > 1e-9
    if int(moving.sum()) < 4:
        return np.nan, np.nan
    xy, v = xy[moving], v[moving]
    hat = v / np.linalg.norm(v, axis=-1, keepdims=True)
    mean_h = hat.mean(axis=0)
    dv = hat - mean_h
    phi = float(np.linalg.norm(mean_h))
    chi = float(len(dv)) * float(max(1.0 - phi**2, 0.0))
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    i, j = np.triu_indices(len(xy), k=1)
    rij = d[i, j]
    cij = (dv[i] * dv[j]).sum(axis=-1)
    if rij.size < 8:
        return np.nan, chi
    edges = np.quantile(rij, np.linspace(0.0, 0.9, n_bins + 1))
    centers, corr = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (rij >= a) & (rij < b)
        if sel.sum() < 4:
            continue
        centers.append(0.5 * (a + b))
        corr.append(float(np.mean(cij[sel])))
    if len(corr) < 2:
        return np.nan, chi
    corr = np.asarray(corr)
    centers = np.asarray(centers)
    c0 = corr[0] if abs(corr[0]) > 1e-9 else 1.0
    target = c0 / np.e
    below = np.flatnonzero(corr <= target)
    if below.size == 0:
        return float(centers[-1]), chi
    k = int(below[0])
    if k == 0:
        return float(centers[0]), chi
    x0, x1 = centers[k - 1], centers[k]
    y0, y1 = corr[k - 1], corr[k]
    frac = (target - y0) / (y1 - y0 + 1e-12)
    return float(x0 + frac * (x1 - x0)), chi


def _lag1_ac(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    x = x - x.mean()
    if x.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _slope(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    t = np.arange(y.size, dtype=np.float64)[m]
    yy = y[m]
    t = t - t.mean()
    den = float(np.dot(t, t))
    if den < 1e-12:
        return 0.0
    return float(np.dot(t, yy - yy.mean()) / den)


def window_feature_dict(
    series: dict[str, np.ndarray],
    positions: np.ndarray,
    velocities: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    """Aggregate one [start, end) window, including expensive window-level terms."""
    feat: dict[str, float] = {}
    for name, arr in series.items():
        sl = arr[start:end]
        feat[f"{name}_mean"] = float(np.nanmean(sl)) if sl.size else np.nan
        feat[f"{name}_std"] = float(np.nanstd(sl)) if sl.size else np.nan
        feat[f"{name}_slope"] = _slope(sl)
    mid = start + (end - start) // 2
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    valid = np.isfinite(pos[mid]).all(axis=-1) & np.isfinite(vel[mid]).all(axis=-1)
    l2k, gk, _ = _graph_spectrum(pos[mid], valid, k=5)
    l2m, gm, nc = _graph_spectrum(pos[mid], valid, radius=150.0)
    clen, chi = _corr_length(pos[mid], vel[mid], valid)
    feat["lambda2_knn"] = l2k
    feat["gap_knn"] = gk
    feat["lambda2_metric"] = l2m
    feat["gap_metric"] = gm
    feat["n_components_metric"] = nc
    feat["corr_length"] = clen
    feat["susceptibility"] = chi
    feat["phi_trans_ac1"] = _lag1_ac(series["phi_trans"][start:end])
    return feat


def iter_bounds(n_frames: int, window_frames: int, hop_frames: int) -> list[tuple[int, int]]:
    w, hop = int(window_frames), int(hop_frames)
    t = int(n_frames)
    if t < w or w <= 0 or hop <= 0:
        return []
    return [(s, s + w) for s in range(0, t - w + 1, hop)]


def extract_windows(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    window_frames: int = 5,
    hop_frames: int = 5,
    names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    names = names or all_window_feature_names()
    bounds = iter_bounds(positions.shape[0], window_frames, hop_frames)
    if not bounds:
        return np.zeros((0, len(names))), names, []
    series = compute_extended_series(positions, velocities)
    rows = []
    for a, b in bounds:
        feat = window_feature_dict(series, positions, velocities, a, b)
        rows.append([feat.get(n, np.nan) for n in names])
    return np.asarray(rows, dtype=np.float64), names, bounds


def columns_for_families(families: Iterable[str], all_names: list[str]) -> list[int]:
    wanted: list[str] = []
    for fam in families:
        wanted.extend(family_feature_names(fam))
    # unique, keep order
    seen = set()
    cols = []
    for n in wanted:
        if n in seen:
            continue
        seen.add(n)
        if n in all_names:
            cols.append(all_names.index(n))
    return cols
