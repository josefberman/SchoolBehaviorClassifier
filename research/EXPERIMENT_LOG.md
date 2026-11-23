# Experiment log

Chronological record of research runs on `research/feature-space-and-sim`.
Numbers are copied from `research/results/*.json` and `research/REPORT.md`.
Nothing here is invented. Annotation files were never edited. Raw H5s are
gitignored (`schooling-datasets/`). Feature caches live in gitignored
`research/cache/`.

Classifier seed is `random_state=42` in `src/classify/grouped.py` unless a
row says otherwise. Grouped CV is leave-one-video-out.

---

## Retroactive — iterations 1–4

Sources: `research/REPORT.md`, `research/OPERATOR_REPORT.md`,
`research/results/audit.json`, `scaleup_lovo.json`, `largen_lovo_w{5,15}_h5.json`,
`sequences_all.json`, `operator_k_sweep.json`, `operator_lovo_{segment,w15_h5,w5_h5}.json`.

### Iteration 1 — leakage-safe audit and physical add-ons

- **Goal.** Replace the published sim-train → real-test score (macro-F1 0.464)
  with grouped LOVO on real videos; test whether extra feature families beat
  the 4 Φ baseline.
- **Change.** `src/features/extended.py`, `src/classify/grouped.py`,
  `scripts/research_{audit,features,sim_realism,plots}.py`. Families: baseline
  4 Φ, fluct, rate (`dlogR`), shape, tunstrom, fields, early, spectral, corr, info.
- **Hyps.** XGB `random_state=42`, `max_depth=4`, `n_estimators=200`,
  `learning_rate=0.08`. Windows hop = window.
- **Results (10-fish LOVO).** w=5 stable5: 4 Φ 0.471, +rate **0.522**. w=15:
  4 Φ 0.524, +rate **0.575**. Non-physical alone 0.320. Source:
  `research/REPORT.md` “Baseline vs best (10-fish LOVO)”.
- **Conclusion.** `dlogR` is the physical add-on. Graph-spectral hurts at N=10.
- **Next.** Scale to 30/70/150; treat transients as sequences; CMA-ES sim fit.

### Iteration 2 — all-N segments, sequences, short CMA

- **Goal.** Repeat dlogR at all N; event-level A→B; joint Couzin CMA-ES.
- **Change.** `scripts/research_{scaleup,sequences,cma_fit}.py`,
  `src/classify/sequences.py`. Segment kinematics from
  `annotation_review/segment_review.csv` (H5s not yet on disk).
- **Results.** Segment stable5 pooled: 4 Φ 0.865 → +dlogR 0.898 → +rate **0.930**
  (`scaleup_lovo.json`). 10-fish HMM events later superseded by all-N run.
  Short CMA overwrote `behaviors_fit` then restored; mixed-N KS target was a
  bug, fixed to 10-fish pools.
- **Conclusion.** dlogR replicates at 30/70 at **segment** scale. Sequences
  over-fire. Short CMA lost to speed-match.
- **Next.** Fetch H5s; window LOVO at large N; longer CMA.

### Iteration 3 — H5 windows at all N, long CMA

- **Goal.** 5-frame and 15-frame LOVO on all 10 H5s; spectral at large N;
  constrained CMA.
- **Change.** `scripts/fetch_trajectories.py`, `research_largen.py`,
  `research_cma_fit.py --maxiter 20 --popsize 12 --seeds 3 --init-from-fit`.
  H5s from `https://huggingface.co/buckets/josefberman/SchoolingFish`.
- **Data.** 591/591 segments. w=5 40 486 windows, w=15 39 304. Video 0200 is
  N=151 (per-N LOVO skipped).
- **Results.** Pooled stable5 w=5: 4 Φ 0.715, +rate 0.748, physical **0.801**.
  w=15: 4 Φ 0.734, physical **0.820**. 10-fish w=15 +rate **0.575** replicates
  iter 1. Spectral helps at 70-fish w=15 (`all` 0.843). Events all-10:
  HMM F1 0.233 vs iid 0.224 vs CUSUM 0.105 (`sequences_all.json`). Long CMA
  mean KS 0.474 → **0.386** (`configs/behaviors_cma_long/`).
- **Bugs.** CMA snapshot overwrote real stills (restored). First long CMA mixed
  all-N real into 10-fish KS (aborted; rerun 10-fish only).
- **Conclusion.** dlogR replicates at windows and at 30/70. Spectral is an
  add-on at N≈70, not a Φ replacement. Sequences not paper-ready.
- **Next.** Josef: drop hand-crafted Φ; try operator-theoretic features.

### Iteration 4 — operator pivot (unscaled kNN, k-sweep)

- **Goal.** Domain-agnostic operator features vs 4 Φ / physical at segment,
  w=15, w=5.
- **Change.** `src/features/operator.py`, `scripts/research_operator.py`.
  First attempt used Delaunay + per-frame median-NND scaling (segment operator
  stable5 **0.479**). Josef: drop Delaunay, stop NND-scaling, signed Hodge,
  unscaled edge-length growth, sweep k then freeze.
- **k-sweep.** `k ∈ {3..9}`, cap N−1. Segment grouped LOVO, full operator
  suite, XGB seed 42. Command: `python scripts/research_operator.py --scale ksweep`.
  Results: `research/results/operator_k_sweep.json`. Frozen **k=8** (pooled
  0.823). Nested inner-LOVO chose k=9 in 6/10 videos. 10-fish prefers k=9
  (0.733 vs 0.624).
- **Three-scale command.** `python scripts/research_operator.py --scale all --skip-ksweep`.
  Results: `operator_lovo_{segment,w15_h5,w5_h5}.json`.
- **Pooled stable5.** operator 0.823 / 0.655 / 0.619 vs 4 Φ 0.929 / 0.734 /
  0.715 vs physical 0.960 / 0.820 / 0.801 (segment / w15 / w=5).
- **Pooled coarse-6 transient F1.** operator 0.740 / 0.290 / 0.286 vs 4 Φ
  0.657 / 0.402 / 0.360.
- **Aborted / failed.** NND-scaled + Delaunay extractor discarded (Josef).
  w=15 LOVO crashed on HGB fine-grained singleton A→B (`early_stopping`
  stratified split); fixed `early_stopping=False`, skipped non-XGB finegrained.
  MLP not used on windows after segment MLP 0.313. torch / LightGBM / CatBoost
  not installed.
- **Conclusion.** Not a Φ replacement. Useful coordinates are `hodge_rad_*`
  and `dyn_dlog_ell`. Do not NND-scale if expansion/compaction matter.
- **Next.** Optimize operators: connection Laplacian, full Hodge 2-complex,
  Kernel-EDMD, Hankel/DMD tuning, joint hyps, duration-constrained events.

---

## 2026-11-23 20:04 UTC — iteration 5 start: optimize operators

- **Goal / hypothesis.** Connection (SO(2) vector diffusion), discrete Hodge
  on kNN triangles, Kernel-EDMD (RBF; no deep Koopman — torch absent), and
  tuned Hankel/DMD can close the gap to 4 Φ (0.929 / 0.734 / 0.715) from the
  k=8 operator baseline (0.823 / 0.655 / 0.619) without writing Φ / M / dlogR.
- **Change.** `src/features/operator.py` families `connection`, `hodge2`,
  `kedmd`, `hankel`; `dyn_dconn_l2`, `dyn_dfrustr`. SO(2) connection Laplacian
  on unscaled headings (Singer 2012) + speed-weighted variant. Helmholtz–Hodge
  of the velocity 1-form on kNN triangles (B1 / B2 Poisson). Kernel-EDMD RBF
  + linear VAMP-2 on GFT velocity stacked with λ₂ and linearly interpolated
  Hodge/connection scalars (not a full geometry at every snapshot). Truncated-SVD
  Hankel-DMD delay/rank grid. `SupervisedHMM.viterbi_min_dwell` expanded-state
  Viterbi. SHAP `TreeExplainer` on segment XGB folds (`src/classify/grouped.py`).
  Eval: `scripts/research_operator_opt.py`.
- **Hyps / grids (segment grouped LOVO, then freeze).**
  - k ∈ {8, 9} (8 is frozen v4 winner; 9 won nested / 10-fish)
  - KEDMD σ multiplier ∈ {0.5, 1.0, 2.0} × median pairwise of window observables
  - KEDMD rank ∈ {4, 8}
  - Hankel delays ∈ {3, 6}
  - Hankel SVD rank ∈ {4, 8}
  - min-dwell = 3 windows for event Viterbi (also decode with unconstrained Viterbi)
  - seed 42
- **Not run.** VAMPNet / deep Koopman encoder — `torch` is not installed.
- **Abandoned / speed fixes (same math, not a new feature family).**
  - First KEDMD sketch recomputed `frame_geometry` at every snapshot (48 geos /
    window). Replaced with GFT series + cheap λ₂ + lerp of Hodge/connection
    scalars so each window still uses 3 full geos (start/mid/end).
  - Dense `lstsq` on the triangle Laplacian (N=151, n_t≈1185) was 0.20 s/frame.
    Switched to scipy sparse CG. Smoke: 150-fish w=15 1.00 s → 0.11 s; 70-fish
    0.15 s → 0.036 s; 10-fish 0.027 s → 0.011 s. All 115 names finite at N=10.
- **Commands.**
  1. `python scripts/research_operator_opt.py --mode sweep`
  2. `python scripts/research_operator_opt.py --mode lovo` (frozen hyps, three scales)
- **Results files (to be written).**
  `research/results/operator_opt_sweep.json`,
  `operator_opt_hyps.json`,
  `operator_opt_lovo_{segment,w15_h5,w5_h5}.json`,
  `operator_opt_three_scales.json`.
- **Status.** Segment sweep finished 2026-11-23 20:16 UTC. Three-scale LOVO
  running after a freeze-column fix (see next entry).

## 2026-11-23 20:16 UTC — iteration 5 segment hyp sweep (freeze)

- **Goal.** Joint grouped-LOVO selection of k, KEDMD (σ, rank), Hankel
  (delay, rank) on whole annotated segments. Freeze one config.
- **Command.** `python3 scripts/research_operator_opt.py --mode all`
  (sweep portion). Seed 42. k ∈ {8, 9}.
- **Results.** `research/results/operator_opt_sweep.json`,
  `operator_opt_hyps.json`, `operator_opt_sweep.csv`.
  Caches: `research/cache/operator_opt_segments_k{8,9}.npz` (gitignored).
- **Pooled stable5 (segment, XGB).** Sources: sweep JSON.

  | cfg | k=8 | k=9 |
  | --- | ---: | ---: |
  | operator_v4 | 0.823 | 0.823 (same 61 cols; k only) |
  | connection alone | 0.325 | 0.336 |
  | hodge2 alone | 0.470 | 0.445 |
  | kedmd alone | 0.442 | 0.425 |
  | hankel grid alone | 0.465 | 0.414 |
  | v4+connection | 0.817 | 0.816 |
  | v4+hodge2 | 0.820 | 0.814 |
  | v4+kedmd (all σ/rank) | 0.805 | 0.817 |
  | kitchen-sink all 115 | 0.803 | 0.815 |
  | base+kedmd_s05_r4 | 0.824 | 0.835 |
  | **base+hdmd_d3_r4** | 0.826 | **0.836** |
  | base+hdmd_d3_r8 | 0.818 | 0.829 |

- **Frozen.** k**=9**, selection `base+hdmd_d3_r4` = v4 + connection + hodge2
  + Hankel delays=3 rank=4 (79 features). Segment stable5 **0.836**.
  Near-tie: `base+kedmd_s05_r4` at k=9 is 0.835. Kitchen-sink 115-col vector
  **loses** (0.815): extra KEDMD/Hankel columns are collinear noise.
- **Conclusion so far.** New families as add-ons, not replacements. Frozen
  set is a small lift over v4 0.823, still far from 4 Φ 0.929.
- **Bug / abort.** First `--mode all` LOVO used the 115-col kitchen-sink as
  `operator_opt` instead of the frozen 79-col subset. Killed after sweep
  write; caches kept. Restart `--mode lovo` with `frozen_cols()`.
- **Next.** Three-scale LOVO at frozen hyps; permutation + SHAP on segment;
  min-dwell events on windows.

## 2026-11-23 20:18–23:02 UTC — iteration 5 three-scale LOVO (frozen hyps)

- **Goal.** Same protocol as iteration 4: grouped LOVO at segment / w=15 /
  w=5 with frozen k=9, Hankel d=3 r=4, plus structural ablations.
- **Command.** `python3 scripts/research_operator_opt.py --mode lovo`
- **Seed.** 42. Frozen: k=9, 79 cols = v4 + connection + hodge2 + hdmd_d3_r4.
- **Results.** `research/results/operator_opt_lovo_{segment,w15_h5,w5_h5}.json`,
  `operator_opt_three_scales.json`, figure
  `research/figures/operator_opt_stable5_three_scales.png`.
- **Pooled stable5 / coarse-6 transient F1** (XGB). Source: those JSONs.

  | abl | stab seg / w15 / w5 | trans seg / w15 / w5 |
  | --- | --- | --- |
  | operator_v4 (k=9) | 0.820 / 0.659 / 0.620 | 0.740 / 0.291 / 0.282 |
  | connection | 0.336 / 0.347 / 0.341 | 0.458 / 0.197 / 0.205 |
  | hodge2 | 0.445 / 0.478 / 0.478 | 0.447 / 0.226 / 0.222 |
  | kedmd | 0.425 / 0.244 / 0.174 | 0.700 / 0.094 / 0.113 |
  | hankel | 0.414 / 0.335 / 0.290 | 0.680 / 0.095 / 0.047 |
  | v4+connection | 0.816 / 0.661 / 0.623 | 0.719 / 0.289 / 0.286 |
  | v4+hodge2 | 0.814 / 0.689 / 0.658 | 0.739 / 0.347 / 0.345 |
  | v4+kedmd | 0.817 / 0.660 / 0.623 | 0.759 / 0.284 / 0.277 |
  | **operator_opt (frozen)** | **0.836 / 0.686 / 0.667** | 0.741 / 0.355 / 0.346 |
  | kitchen-sink 115 | 0.815 / 0.692 / 0.667 | 0.764 / 0.359 / 0.346 |
  | 4 Φ | 0.929 / 0.734 / 0.715 | 0.657 / 0.402 / 0.360 |
  | physical | 0.960 / 0.820 / 0.801 | 0.852 / 0.516 / 0.482 |
  | opt+physical | 0.957 / 0.821 / 0.802 | 0.864 / 0.523 / 0.485 |

- **Per-N stable5 operator_opt vs v4 vs 4 Φ.** 150-fish skipped (one video).

  | scale | 10-fish opt / v4 / Φ | 30-fish | 70-fish |
  | --- | --- | --- | --- |
  | segment | 0.708 / 0.733 / 0.744 | 0.702 / 0.676 / 0.886 | 0.668 / 0.677 / 0.905 |
  | w=15 | 0.376 / 0.390 / 0.524 | 0.571 / 0.530 / 0.733 | 0.652 / 0.632 / 0.794 |
  | w=5 | 0.351 / 0.377 / 0.471 | 0.537 / 0.488 / 0.697 | 0.637 / 0.595 / 0.779 |

- **Classifiers (frozen opt, pooled stable5).** segment XGB 0.836 / HGB 0.812 /
  logreg 0.793. w=15 XGB 0.686 / HGB 0.671. w=5 XGB 0.667 / HGB 0.650.
- **Permutation (segment, 3 shuffles).** `dyn_dlog_ell` 0.124, `hodge_rad_mean`
  0.079, `dmd_radius` 0.062, `h2_grad_frac` 0.059, `hdmd_freq_max` 0.043.
- **SHAP |value| mean (segment TreeExplainer).** `dyn_dlog_ell` 0.591,
  `h2_grad_frac` 0.460, `hodge_rad_mean` 0.392, `mag_q25_dl2` 0.269,
  `dmd_radius` 0.236.
- **Events ±45 frames, min_dwell=3.** w=15: HMM 0.168, min-dwell 0.168, iid
  0.172 (836 / 847 / 1500 pred vs 248 true). w=5: HMM 0.211, min-dwell 0.211,
  iid 0.189 (699 / 701 / 1413 pred). Duration constraint does not cut FPs.
- **Conclusion.** Frozen operators 0.836 / 0.686 / 0.667 vs 4 Φ 0.929 /
  0.734 / 0.715. Lift over v4 is +0.016 / +0.027 / +0.047, almost all from
  the Helmholtz–Hodge 2-complex (`v4+hodge2` 0.814 / 0.689 / 0.658).
  Connection is ~0.34 alone and does not help v4. Kernel-EDMD collapses at
  short windows (0.174 at w=5). Kitchen-sink loses at segment. 10-fish is
  worse with the pooled freeze than with v4. min-dwell Viterbi is not an
  event detector. Still not a Φ replacement.
- **Next.** Stop adding Koopman/EDMD columns. If another operator pass: keep
  unscaled Hodge 1-form + 2-complex + `dyn_dlog_ell`, drop KEDMD, retune k
  on 10-fish separately. VAMPNet still blocked (`torch` absent).
