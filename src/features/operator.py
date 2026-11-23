"""Permutation-invariant operator-theoretic window features.

Unscaled kNN only (no Delaunay, no per-frame NND / σ normalization).
Positions are COM-centered for translation invariance; distances and
velocities stay in native units so expansion/compaction keep their scale.
Fish indices never enter the feature vector.

Families
--------
spatial   — unscaled kNN normalized + combinatorial Laplacian spectra, edge lengths
magnetic  — Hermitian magnetic Laplacian (unscaled velocity as phase)
hodge     — signed divergence / radial flux + curl of the velocity 1-form
dyn       — spectral change + kNN edge-length growth across the window
dmd       — Exact DMD + Hankel-DMD on graph-Fourier coefficients of raw velocity
koopman   — DMD residual + HAVOK forcing of λ₂(t)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.linalg import pinv

N_EIG = 6
R_GFT = 4
HEAT_T = (0.25, 1.0, 4.0)
MAG_Q = (0.25, 0.5)
# Default before the k-sweep; scripts/research_operator.py overwrites this
# with the frozen argmax of leakage-safe segment LOVO (see operator_k_sweep.json).
K_NN = 8  # frozen from leakage-safe segment LOVO k-sweep; see research/results/operator_k_sweep.json
K_CANDIDATES = (3, 4, 5, 6, 7, 8, 9)


def resolve_k(k: int | None, n_valid: int) -> int:
    """kNN degree, capped at N-1 (relevant for 10-fish when k=9)."""
    kk = int(K_NN if k is None else k)
    return int(min(max(kk, 1), max(int(n_valid) - 1, 1)))


def _nanfeat() -> dict[str, float]:
    return {n: np.nan for n in all_operator_names()}


def spatial_names() -> list[str]:
    names = [
        "spat_knn_l2",
        "spat_knn_l2_rel",
        "spat_knn_gap",
        "spat_knn_lmax",
    ]
    names.extend([f"spat_knn_l{k}" for k in range(3, N_EIG + 1)])
    names.extend(
        [
            "spat_knn_m2",
            "spat_knn_m3",
            "spat_knn_entropy",
            "spat_knn_heat_025",
            "spat_knn_heat_1",
            "spat_knn_heat_4",
            "spat_knn_ncomp",
            "spat_knn_l2_comb",
            "spat_knn_ell_mean",
            "spat_knn_ell_med",
        ]
    )
    return names


def magnetic_names() -> list[str]:
    names = []
    for q in ("q25", "q50"):
        names.extend(
            [
                f"mag_{q}_l2",
                f"mag_{q}_gap",
                f"mag_{q}_heat1",
                f"mag_{q}_dl2",
                f"mag_{q}_entropy",
            ]
        )
    return names


def hodge_names() -> list[str]:
    return [
        "hodge_grad",
        "hodge_div_mean",
        "hodge_div_std",
        "hodge_rad_mean",
        "hodge_rad_std",
        "hodge_curl",
        "hodge_frac_curl",
        "hodge_energy",
    ]


def dyn_names() -> list[str]:
    return [
        "dyn_dl2",
        "dyn_dgap",
        "dyn_dheat",
        "dyn_spec_w1",
        "dyn_fiedler_corr",
        "dyn_knn_jaccard",
        "dyn_dell",
        "dyn_dlog_ell",
    ]


def dmd_names() -> list[str]:
    return [
        "dmd_growth_max",
        "dmd_growth_mean",
        "dmd_growth_std",
        "dmd_freq_max",
        "dmd_freq_wmean",
        "dmd_radius",
        "dmd_resid",
        "dmd_mode_conc",
        "hdmd_growth_max",
        "hdmd_freq_max",
        "hdmd_resid",
    ]


def koopman_names() -> list[str]:
    return [
        "koop_resid_mean",
        "koop_resid_max",
        "havok_vr_absmean",
        "havok_vr_kurt",
        "havok_vr_tail",
        "havok_energy_ratio",
    ]


def all_operator_names() -> list[str]:
    return spatial_names() + magnetic_names() + hodge_names() + dyn_names() + dmd_names() + koopman_names()


FAMILY_NAMES = {
    "spatial": spatial_names(),
    "magnetic": magnetic_names(),
    "hodge": hodge_names(),
    "dyn": dyn_names(),
    "dmd": dmd_names(),
    "koopman": koopman_names(),
    "operator": all_operator_names(),
}


def columns_for_operator_families(families: Iterable[str], all_names: list[str]) -> list[int]:
    wanted: list[str] = []
    for fam in families:
        wanted.extend(FAMILY_NAMES[fam])
    seen = set()
    cols = []
    for n in wanted:
        if n in seen or n not in all_names:
            continue
        seen.add(n)
        cols.append(all_names.index(n))
    return cols


def _valid_mask(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return np.isfinite(pos).all(axis=-1) & np.isfinite(vel).all(axis=-1)


def _pairwise(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def _knn_adj(d: np.ndarray, k: int) -> np.ndarray:
    m = d.shape[0]
    kk = min(max(k, 1), m - 1)
    nn = np.argpartition(d, kth=kk - 1, axis=-1)[:, :kk]
    adj = np.zeros((m, m), dtype=np.float64)
    rows = np.repeat(np.arange(m), kk)
    adj[rows, nn.ravel()] = 1.0
    return np.maximum(adj, adj.T)


def _inv_dist_weights(adj: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Unscaled inverse-distance weights on the kNN support."""
    dd = np.where(np.isfinite(d), np.maximum(d, 1e-9), np.inf)
    w = adj / dd
    w[~np.isfinite(w)] = 0.0
    np.fill_diagonal(w, 0.0)
    return w


def _norm_lap(w: np.ndarray) -> np.ndarray:
    deg = np.maximum(w.sum(axis=1), 1e-12)
    dm = np.diag(deg ** (-0.5))
    return np.eye(w.shape[0]) - dm @ w @ dm


def _comb_lap(w: np.ndarray) -> np.ndarray:
    deg = w.sum(axis=1)
    return np.diag(deg) - w


def _spectrum_summaries(evals: np.ndarray, prefix: str) -> dict[str, float]:
    ev = np.sort(np.real(evals))
    ev = ev[ev > -1e-8]
    out: dict[str, float] = {}
    nontrivial = ev[1:] if ev.size > 1 else np.array([])
    l2 = float(nontrivial[0]) if nontrivial.size else np.nan
    gap = float(nontrivial[1] - nontrivial[0]) if nontrivial.size > 1 else np.nan
    lmax = float(ev[-1]) if ev.size else np.nan
    out[f"{prefix}_l2"] = l2
    out[f"{prefix}_l2_rel"] = float(l2 / lmax) if np.isfinite(l2) and np.isfinite(lmax) and lmax > 1e-12 else np.nan
    out[f"{prefix}_gap"] = gap
    out[f"{prefix}_lmax"] = lmax
    for k in range(3, N_EIG + 1):
        out[f"{prefix}_l{k}"] = float(nontrivial[k - 2]) if nontrivial.size >= k - 1 else np.nan
    if ev.size:
        p = np.clip(ev, 0, None)
        s = p.sum()
        p = p / s if s > 1e-12 else np.full_like(p, 1.0 / max(p.size, 1))
        out[f"{prefix}_m2"] = float(np.mean(ev**2))
        out[f"{prefix}_m3"] = float(np.mean(ev**3))
        out[f"{prefix}_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
        for t, tag in zip(HEAT_T, ("025", "1", "4")):
            out[f"{prefix}_heat_{tag}"] = float(np.mean(np.exp(-t * np.clip(ev, 0, None))))
        out[f"{prefix}_ncomp"] = float(np.sum(ev < 1e-6))
    else:
        for k in ("m2", "m3", "entropy", "heat_025", "heat_1", "heat_4", "ncomp"):
            out[f"{prefix}_{k}"] = np.nan
    return out


def _magnetic_weight(w: np.ndarray, xy: np.ndarray, vel: np.ndarray, q: float) -> np.ndarray:
    e = xy[None, :, :] - xy[:, None, :]
    vavg = 0.5 * (vel[:, None, :] + vel[None, :, :])
    cross = e[..., 0] * vavg[..., 1] - e[..., 1] * vavg[..., 0]
    elen = np.linalg.norm(e, axis=-1) + 1e-12
    theta = np.arctan2(cross, elen * (np.linalg.norm(vavg, axis=-1) + 1e-12))
    np.fill_diagonal(theta, 0.0)
    W = w * np.exp(1j * (2.0 * np.pi * q) * theta)
    W = 0.5 * (W + W.conj().T)
    np.fill_diagonal(W, 0.0)
    return W


def _norm_mag_lap(W: np.ndarray, deg: np.ndarray) -> np.ndarray:
    d = np.maximum(deg, 1e-12)
    dm = np.diag(d ** (-0.5))
    return np.eye(W.shape[0]) - dm @ W @ dm


def _hodge(xy: np.ndarray, vel: np.ndarray, adj: np.ndarray) -> dict[str, float]:
    """Signed 1-form Hodge energies on unscaled kNN edges (COM-centered xy)."""
    empty = {k: np.nan for k in hodge_names()}
    i, j = np.triu_indices(xy.shape[0], k=1)
    mask = adj[i, j] > 0
    if not np.any(mask):
        return empty
    i, j = i[mask], j[mask]
    e = xy[j] - xy[i]
    elen = np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12
    ehat = e / elen
    vavg = 0.5 * (vel[i] + vel[j])
    # signed stretch along the edge (positive ⇒ edges lengthening / local expansion)
    alpha = np.sum(vavg * ehat, axis=-1)
    # signed radial flux relative to COM
    mid = 0.5 * (xy[i] + xy[j])
    rlen = np.linalg.norm(mid, axis=-1, keepdims=True) + 1e-12
    rhat = mid / rlen
    rad = np.sum(vavg * rhat, axis=-1)
    omega = vavg[:, 0] * ehat[:, 1] - vavg[:, 1] * ehat[:, 0]
    eg = float(np.mean(alpha**2))
    ec = float(np.mean(omega**2))
    tot = eg + ec
    return {
        "hodge_grad": eg,
        "hodge_div_mean": float(np.mean(alpha)),
        "hodge_div_std": float(np.std(alpha)),
        "hodge_rad_mean": float(np.mean(rad)),
        "hodge_rad_std": float(np.std(rad)),
        "hodge_curl": ec,
        "hodge_frac_curl": float(ec / tot) if tot > 1e-12 else np.nan,
        "hodge_energy": tot,
    }


def _dmd(X: np.ndarray, dt: float = 1.0) -> dict[str, float]:
    keys = [
        "growth_max",
        "growth_mean",
        "growth_std",
        "freq_max",
        "freq_wmean",
        "radius",
        "resid",
        "mode_conc",
    ]
    empty = {k: np.nan for k in keys}
    if X.ndim != 2 or X.shape[1] < 3 or X.shape[0] < 1:
        return empty
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0)
    X0, X1 = X[:, :-1], X[:, 1:]
    try:
        A = X1 @ pinv(X0)
        mu = np.linalg.eigvals(A)
    except Exception:
        return empty
    mu = mu[np.isfinite(mu) & (np.abs(mu) > 1e-12)]
    if mu.size == 0:
        return empty
    dt = float(max(dt, 1e-9))
    logmu = np.log(mu) / dt
    growth = np.real(logmu)
    ang = np.abs(np.angle(mu))
    keep = ang < 0.9 * np.pi
    freq = np.abs(np.imag(logmu))
    freq_use = freq[keep] if np.any(keep) else freq
    amp = np.abs(mu)
    w_use = (amp[keep] / max(amp[keep].sum(), 1e-12)) if np.any(keep) else amp / max(amp.sum(), 1e-12)
    pred = A @ X0
    resid = float(np.linalg.norm(X1 - pred) / (np.linalg.norm(X1) + 1e-12))
    evA = np.abs(np.linalg.eigvals(A))
    conc = float(np.max(evA) / (np.sum(evA) + 1e-12)) if evA.size else np.nan
    return {
        "growth_max": float(np.max(growth)),
        "growth_mean": float(np.mean(growth)),
        "growth_std": float(np.std(growth)),
        "freq_max": float(np.max(freq_use)) if freq_use.size else np.nan,
        "freq_wmean": float(np.sum(w_use * freq_use)) if freq_use.size else np.nan,
        "radius": float(np.max(np.abs(mu))),
        "resid": resid,
        "mode_conc": conc,
    }


def _hankel(sig: np.ndarray, delays: int) -> np.ndarray:
    d, T = sig.shape
    if T < delays + 2:
        return np.zeros((0, 0))
    cols = T - delays + 1
    H = np.zeros((d * delays, cols), dtype=np.float64)
    for i in range(delays):
        H[i * d : (i + 1) * d, :] = sig[:, i : i + cols]
    return H


def _havok_stats(x: np.ndarray, delays: int = 4) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    empty = {k: np.nan for k in ("vr_absmean", "vr_kurt", "vr_tail", "energy_ratio")}
    if x.size < delays + 3:
        return empty
    H = _hankel(x[None, :], delays)
    if H.size == 0 or H.shape[1] < 3:
        return empty
    try:
        _, S, Vt = np.linalg.svd(H, full_matrices=False)
    except Exception:
        return empty
    vr = Vt[-1]
    vr = vr - vr.mean()
    s2 = float(np.mean(vr**2)) + 1e-12
    energy = S**2
    return {
        "vr_absmean": float(np.mean(np.abs(vr))),
        "vr_kurt": float(np.mean(vr**4) / s2**2),
        "vr_tail": float(np.mean(np.abs(vr) > 2.0 * np.sqrt(s2))),
        "energy_ratio": float(energy[-1] / (energy.sum() + 1e-12)),
    }


def _wasserstein_spec(a: np.ndarray, b: np.ndarray) -> float:
    def cdf(ev):
        ev = np.sort(np.real(ev))
        ev = ev[(ev >= -1e-8) & (ev <= 2.5)]
        if ev.size < 2:
            return np.linspace(0, 2, 16), np.linspace(0, 1, 16)
        grid = np.linspace(0, 2, 16)
        return grid, np.searchsorted(ev, grid) / ev.size

    _, fa = cdf(a)
    _, fb = cdf(b)
    return float(np.mean(np.abs(fa - fb)))


def frame_geometry(pos: np.ndarray, vel: np.ndarray, k: int | None = None) -> dict | None:
    """Unscaled kNN graph, spectra, signed Hodge, GFT basis. COM-centered xy."""
    valid = _valid_mask(pos, vel)
    idx = np.flatnonzero(valid)
    m = int(idx.size)
    if m < 4:
        return None
    xy = np.asarray(pos[idx], dtype=np.float64)
    vv = np.asarray(vel[idx], dtype=np.float64)
    xy = xy - xy.mean(axis=0)
    d = _pairwise(xy)
    k = resolve_k(k, m)
    knn = _knn_adj(d, k)
    w = _inv_dist_weights(knn, d)
    L = _norm_lap(w)
    evals, evecs_k = np.linalg.eigh(L)
    evals_c = np.linalg.eigvalsh(_comb_lap(w))
    ell = d[knn > 0]
    ell = ell[np.isfinite(ell)]
    feat: dict[str, float] = {}
    feat.update(_spectrum_summaries(evals, "spat_knn"))
    evc = np.sort(np.real(evals_c))
    evc = evc[evc > -1e-8]
    feat["spat_knn_l2_comb"] = float(evc[1]) if evc.size > 1 else np.nan
    feat["spat_knn_ell_mean"] = float(np.mean(ell)) if ell.size else np.nan
    feat["spat_knn_ell_med"] = float(np.median(ell)) if ell.size else np.nan
    deg = w.sum(axis=1)
    for q, tag in zip(MAG_Q, ("q25", "q50")):
        W = _magnetic_weight(w, xy, vv, q)
        evm = np.linalg.eigvalsh(_norm_mag_lap(W, deg))
        sm = _spectrum_summaries(evm, f"mag_{tag}")
        feat[f"mag_{tag}_l2"] = sm[f"mag_{tag}_l2"]
        feat[f"mag_{tag}_gap"] = sm[f"mag_{tag}_gap"]
        feat[f"mag_{tag}_heat1"] = sm[f"mag_{tag}_heat_1"]
        feat[f"mag_{tag}_entropy"] = sm[f"mag_{tag}_entropy"]
        feat[f"mag_{tag}_dl2"] = (
            float(sm[f"mag_{tag}_l2"] - feat["spat_knn_l2"])
            if np.isfinite(sm[f"mag_{tag}_l2"]) and np.isfinite(feat["spat_knn_l2"])
            else np.nan
        )
    feat.update(_hodge(xy, vv, knn))
    order = np.argsort(evals)
    evecs = evecs_k[:, order]
    r = min(R_GFT, m - 1)
    gft = evecs[:, 1 : 1 + r]
    return {
        "feat": feat,
        "evals_knn": evals,
        "evecs_gft": gft,
        "idx": idx,
        "knn": knn,
        "xy": xy,
        "vv": vv,
        "l2": feat["spat_knn_l2"],
        "ell": feat["spat_knn_ell_mean"],
        "fiedler": evecs[:, 1] if m > 1 else None,
    }


def _snapshot_frames(start: int, end: int, max_snapshots: int = 48) -> tuple[np.ndarray, float]:
    T = int(end - start)
    if T <= 0:
        return np.zeros(0, dtype=np.int32), 1.0
    if T <= max_snapshots:
        return np.arange(start, end, dtype=np.int32), 1.0
    frames = np.unique(np.linspace(start, end - 1, max_snapshots, dtype=np.int32))
    dt = float(np.median(np.diff(frames))) if frames.size > 1 else 1.0
    return frames, max(dt, 1.0)


def _gft_series(pos: np.ndarray, vel: np.ndarray, frames: np.ndarray, geo_mid: dict) -> np.ndarray:
    idx = geo_mid["idx"]
    U = geo_mid["evecs_gft"]
    r = U.shape[1]
    C = np.zeros((r * 2, int(frames.size)), dtype=np.float64)
    for t, fr in enumerate(frames):
        xy = np.asarray(pos[int(fr), idx], dtype=np.float64)
        vv = np.asarray(vel[int(fr), idx], dtype=np.float64)
        if not np.isfinite(xy).all() or not np.isfinite(vv).all():
            continue
        cx = U.T @ vv[:, 0]
        cy = U.T @ vv[:, 1]
        C[:, t] = np.concatenate([cx, cy])
    return C


def lambda2_knn(pos: np.ndarray, vel: np.ndarray, k: int | None = None) -> float:
    """Cheap Fiedler value of the unscaled inverse-distance kNN Laplacian."""
    valid = _valid_mask(pos, vel)
    idx = np.flatnonzero(valid)
    m = int(idx.size)
    if m < 4:
        return np.nan
    xy = np.asarray(pos[idx], dtype=np.float64)
    xy = xy - xy.mean(axis=0)
    d = _pairwise(xy)
    w = _inv_dist_weights(_knn_adj(d, resolve_k(k, m)), d)
    ev = np.sort(np.real(np.linalg.eigvalsh(_norm_lap(w))))
    return float(ev[1]) if ev.size > 1 else np.nan


def window_operator_features(
    pos: np.ndarray,
    vel: np.ndarray,
    start: int,
    end: int,
    cache: dict[int, dict] | None = None,
    k: int | None = None,
) -> dict[str, float]:
    """Operator features on [start, end). Missing pieces are NaN."""
    out = _nanfeat()
    T = end - start
    if T < 2:
        return out

    def geo_at(fr: int):
        if cache is not None and fr in cache:
            return cache[fr]
        g = frame_geometry(pos[fr], vel[fr], k=k)
        if cache is not None:
            cache[fr] = g
        return g

    mid = start + T // 2
    geo_mid = geo_at(mid)
    if geo_mid is None:
        return out
    out.update(geo_mid["feat"])
    geo0 = geo_at(start)
    geo1 = geo_at(end - 1)
    dt_win = float(max(T - 1, 1))
    if geo0 is not None and geo1 is not None:
        out["dyn_dl2"] = (
            float(geo1["l2"] - geo0["l2"]) if np.isfinite(geo1["l2"]) and np.isfinite(geo0["l2"]) else np.nan
        )
        g0, g1 = geo0["feat"].get("spat_knn_gap", np.nan), geo1["feat"].get("spat_knn_gap", np.nan)
        out["dyn_dgap"] = float(g1 - g0) if np.isfinite(g0) and np.isfinite(g1) else np.nan
        h0, h1 = geo0["feat"].get("spat_knn_heat_1", np.nan), geo1["feat"].get("spat_knn_heat_1", np.nan)
        out["dyn_dheat"] = float(h1 - h0) if np.isfinite(h0) and np.isfinite(h1) else np.nan
        out["dyn_spec_w1"] = _wasserstein_spec(geo0["evals_knn"], geo1["evals_knn"])
        f0, f1 = geo0["fiedler"], geo1["fiedler"]
        if f0 is not None and f1 is not None and f0.size == f1.size:
            c = float(np.dot(f0, f1) / (np.linalg.norm(f0) * np.linalg.norm(f1) + 1e-12))
            out["dyn_fiedler_corr"] = abs(c)
        k0, k1 = geo0["knn"], geo1["knn"]
        if k0.shape == k1.shape:
            a = k0 > 0
            b = k1 > 0
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            out["dyn_knn_jaccard"] = float(inter / union) if union else np.nan
        e0, e1 = geo0["ell"], geo1["ell"]
        if np.isfinite(e0) and np.isfinite(e1) and e0 > 1e-12 and e1 > 1e-12:
            out["dyn_dell"] = float((e1 - e0) / dt_win)
            out["dyn_dlog_ell"] = float((np.log(e1) - np.log(e0)) / dt_win)
    frames, dt = _snapshot_frames(start, end, max_snapshots=48)
    C = _gft_series(pos, vel, frames, geo_mid)
    dmd = _dmd(C, dt=dt)
    out["dmd_growth_max"] = dmd["growth_max"]
    out["dmd_growth_mean"] = dmd["growth_mean"]
    out["dmd_growth_std"] = dmd["growth_std"]
    out["dmd_freq_max"] = dmd["freq_max"]
    out["dmd_freq_wmean"] = dmd["freq_wmean"]
    out["dmd_radius"] = dmd["radius"]
    out["dmd_resid"] = dmd["resid"]
    out["dmd_mode_conc"] = dmd["mode_conc"]
    delays = 3 if C.shape[1] >= 8 else 2
    H = _hankel(C, delays)
    hd = _dmd(H, dt=dt) if H.size else {k: np.nan for k in dmd}
    out["hdmd_growth_max"] = hd.get("growth_max", np.nan)
    out["hdmd_freq_max"] = hd.get("freq_max", np.nan)
    out["hdmd_resid"] = hd.get("resid", np.nan)
    out["koop_resid_mean"] = dmd["resid"]
    if C.shape[1] >= 3 and np.isfinite(C).all():
        X0, X1 = C[:, :-1], C[:, 1:]
        try:
            A = X1 @ pinv(X0)
            step = np.linalg.norm(X1 - A @ X0, axis=0) / (np.linalg.norm(X1, axis=0) + 1e-12)
            out["koop_resid_mean"] = float(np.mean(step))
            out["koop_resid_max"] = float(np.max(step))
        except Exception:
            out["koop_resid_max"] = dmd["resid"]
    else:
        out["koop_resid_max"] = dmd["resid"]
    l2_series = []
    l2_frames, _ = _snapshot_frames(start, end, max_snapshots=24)
    for fr in l2_frames:
        fr = int(fr)
        if cache is not None and cache.get(fr) is not None:
            l2_series.append(cache[fr]["l2"])
        else:
            l2_series.append(lambda2_knn(pos[fr], vel[fr], k=k))
    hv = _havok_stats(np.asarray(l2_series, dtype=np.float64), delays=min(4, max(2, len(l2_series) // 2)))
    out["havok_vr_absmean"] = hv["vr_absmean"]
    out["havok_vr_kurt"] = hv["vr_kurt"]
    out["havok_vr_tail"] = hv["vr_tail"]
    out["havok_energy_ratio"] = hv["energy_ratio"]
    return out
