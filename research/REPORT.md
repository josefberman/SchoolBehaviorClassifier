# Feature space and simulation realism

Josef Berman, Swarm & AI Lab (University of Haifa). Research branch
`research/feature-space-and-sim`. Annotation files were not modified.

Every number below comes from a script in this branch. JSON dumps live in
`research/results/`. Figures live in `research/figures/`. Chronological
run log (hyps, commands, seeds, failed attempts): [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## Questions

1. With a **leakage-safe** protocol (leave-one-video-out), how well do the four
   published order parameters classify the five stable CouzinLab behaviours and
   the transients between them — at 10 fish **and** at 30/70/150?
2. Which **new feature families** — physical or not — move macro-F1, especially
   on expansion/compaction and on transients? Does `dlogR` / w=15 replicate
   when N grows, and do graph-spectral features start helping?
3. Can transients be recovered as **change-points / sequences** (event F1 with
   a tolerance window, plus A→B identity) better than i.i.d. windows?
4. Why do YAML simulations **not look like** the real videos, and can a **joint**
   Couzin calibration (CMA-ES on the KS vector) match NND and R_gyr without
   collapsing milling?

## Data and labels (audit)

Source: `annotations/*.csv` (10 videos, **591 segments**). Transients are
canonical labels containing `_to_`. No unknown labels. **No overlapping
segments.** **195 gaps** between consecutive annotations (median **4.50 s**);
those frames are unlabeled, not relabeled.

| Kind | Segments | Dominant classes |
| --- | ---: | --- |
| Stable | 343 | traveling 165, compaction 54, shoaling 51, expansion 38, milling 35 |
| Transient | 248 | traveling_to_compaction 38, traveling_to_shoaling 36, shoaling_to_traveling 34, expansion_to_traveling 30, traveling_to_milling / milling_to_traveling 21 each; 11 other `A_to_B` labels with n ≤ 14 |

**Trajectory coverage.** CouzinLab H5s are gitignored. Iteration 3 fetched
them from `https://huggingface.co/buckets/josefberman/SchoolingFish` into
`schooling-datasets/` (`scripts/fetch_trajectories.py`). All **10** annotated
videos load; **591 / 591** official segments have trajectories. Video 0200 is
stored as **N = 151**. Window counts (hop = 5): w=5 **40 486**; w=15 **39 304**.
150-fish is one video, so per-N LOVO is skipped; it is included in pooled.

**Iteration 2 substitute (kept).** Before H5s arrived, official labels were
joined to `annotation_review/segment_review.csv` (569 / 591 segments,
segment-level). That table is not used for the window numbers below.

`annotated_config.csv` and `annotated_features.csv` cover all 10 videos
(inverse-fitted Couzin weights and 3-Φ summaries). Those tables are used as a
second, trajectory-free protocol — not as a substitute for the H5 windows.

## Evaluation protocol

**Published protocol (kept, not replaced).** Sim-train → real-test, 5-class,
no transients (`results/real_eval_metrics.json`):

| | n | acc | bal-acc | macro-F1 | T | M | S | E+ | E− |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sim test | 14 520 | — | — | **0.870** | — | — | — | — | — |
| Real (sim-trained XGB) | 35 654 | 0.730 | 0.588 | **0.464** | 0.874 | 0.515 | 0.497 | 0.167 | 0.269 |

That number mixes **domain shift** with class imbalance. It is not a real-data
CV score.

**This branch.** Leave-one-video-out grouped CV (`src/classify/grouped.py`),
class-weighted XGB, non-overlapping windows (hop = window). Adjacent 5-frame
windows from the same clip are dependent; a random split would leak. Josef’s
SpectralSchooling 118-D fingerprint XGB (macro-F1 ≈ 0.85, random 80/20) has
the same videos in train and test and is **not** leakage-safe.

Four tasks:

- `stable5` — traveling / milling / shoaling / expansion / compaction
- `coarse6` — stable5 + one `transient` superclass
- `finegrained` — all `A_to_B` labels present
- `binary_transient` — stable vs any transient

## Baseline vs best (10-fish LOVO)

### Stable 5-class

| Protocol | Features | macro-F1 | bal-acc | T | M | S | E+ | E− |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w=5, 4 Φ | baseline | 0.471 | 0.564 | 0.981 | 0.000 | 0.679 | 0.362 | 0.332 |
| w=5 | **baseline+rate** | **0.522** | 0.572 | 0.987 | 0.000 | 0.694 | **0.544** | 0.385 |
| w=5 | physical (7 families) | 0.518 | 0.514 | 0.991 | 0.000 | 0.688 | 0.497 | 0.416 |
| w=5 | all families | 0.489 | 0.483 | 0.991 | 0.000 | 0.642 | 0.456 | 0.357 |
| w=5 | nonphysical only | 0.320 | 0.332 | 0.984 | 0.000 | 0.428 | 0.071 | 0.117 |
| w=15, 4 Φ | baseline | 0.524 | 0.572 | 0.988 | 0.000 | 0.790 | 0.457 | 0.385 |
| w=15 | **baseline+rate** | **0.575** | **0.605** | 0.991 | 0.000 | **0.850** | **0.618** | 0.419 |
| w=15 | physical | 0.543 | 0.532 | 0.993 | 0.000 | 0.730 | 0.545 | 0.448 |
| w=15 | all | 0.542 | 0.530 | 0.993 | 0.000 | 0.719 | 0.565 | 0.434 |
| all 10 videos | inferred Couzin weights | 0.469 | 0.519 | 0.878 | **0.544** | 0.631 | 0.137 | 0.153 |
| all 10 videos | segment 3-Φ (`annotated_features.csv`) | 0.586 | 0.589 | 0.994 | 0.983 | 0.952 | 0.000 | 0.000 |

n (w=5 stable) = 12 635 windows, 3 videos. n (w=15 stable) = 12 497.
n (inferred weights) = 11 796 windows, 10 videos.

**Best leakage-safe 10-fish result:** 15-frame **baseline+rate**, macro-F1
**0.575** vs 4-Φ **0.524** (Δ **+0.051**). Expansion F1 **0.457 → 0.618**.
Shoaling **0.790 → 0.850**. Compaction **0.385 → 0.419**. Milling stays 0
because it is absent from two of three videos.

The 3-Φ segment table looks strong on T/M/S because those Φ *are* how the
states were named, and expansion/compaction have n = 5 / 3 segments in that
CSV, so LOVO F1 is 0. It is not an independent test set.

### Transients

| Task | Features | overall macro-F1 | transient F1 (or transient-class macro) |
| --- | --- | ---: | ---: |
| coarse6, w=5 | baseline | 0.363 | 0.289 |
| coarse6, w=5 | physical | 0.404 | 0.364 |
| coarse6, w=15 | baseline | 0.384 | 0.316 |
| coarse6, w=15 | all | **0.449** | **0.383** |
| binary, w=5 | baseline | 0.618 | 0.311 |
| binary, w=5 | physical | 0.635 | 0.313 |
| binary, w=15 | baseline | 0.631 | 0.321 |
| binary, w=15 | all | 0.633 | 0.297 |
| finegrained, w=5 | baseline | 0.129 | 0.040 |
| finegrained, w=15 | all | 0.170 | 0.064 |

Fine-grained `A_to_B` labels are not a 5- or 15-frame problem: many classes
have < 20 windows and are video-specific. Binary transient detection is
mediocre (F1 ≈ 0.31) and **rate features do not rescue it**. Collapsing to one
`transient` class is the only setting where extra families help (0.316 →
0.383 at w=15).

Figures: `research/figures/ablation_stable5_w15_h5.png`,
`cm_stable5_baseline+rate_w15_h5.png`, `imp_stable5_physical_w15_h5.png`.

## Feature families

Implemented in `src/features/extended.py`. Original 5-frame Φ path
(`src/features/windows.py`, `WINDOW_FRAMES = 5`) is unchanged.

| Family | Contents | Role |
| --- | --- | --- |
| baseline | mean Φ_trans, Φ_tan, Φ_rad±, Φ_local | published 4-D space |
| fluct | window std of the four Φ | already computed, previously dropped |
| **rate** | slopes of the four Φ + **d log R / dt** | expansion / transients |
| shape | speed, CV, NND, R_gyr, aspect, density, COM speed | scale |
| tunstrom | unsigned rotation M, local–global mismatch | milling |
| fields | COM-frame vorticity, divergence, front–back density | hydro-like |
| early | lag-1 autocorrelation of Φ_trans | critical slowing down |
| spectral | Laplacian λ₂ / gap on kNN and metric graphs | non-physical |
| corr | connected velocity correlation length, susceptibility | Cavagna |
| info | heading-histogram entropy | information-theoretic |

### What matters (permutation Δ macro-F1, w=15 physical)

1. `phi_trans_mean` **0.323**
2. **`dlogR_mean` 0.124**
3. `phi_rad_pm_mean` 0.087
4. `phi_local_mean` 0.039
5. `com_speed_mean` 0.031
6. `nnd_mean` 0.021

On the full feature set the same ranking holds, then `heading_entropy_mean`
(0.069) and `susceptibility` (0.022). Graph-spectral features never enter the
top five at N = 10; `baseline+spectral` **hurts** stable5 (w=5: 0.471 → 0.441).

`dlogR` is the signed relative expansion rate of the gyration radius. Φ_rad±
is a *heading* radial order; a compacting school can still point inward after
the radius has stopped shrinking. The two are not redundant, which is why
expansion F1 jumps when `dlogR` is added and why permuting it costs more than
permuting Φ_rad±.

### Ablation summary (stable5, w=5)

| Add-on to baseline | Δ macro-F1 |
| --- | ---: |
| rate (`dlogR` + Φ slopes) | **+0.051** |
| fields | +0.030 |
| shape | +0.021 |
| fluct | +0.014 |
| corr / info / tunstrom | ≤ +0.004 |
| early (AC1) | −0.001 |
| spectral | **−0.030** |
| nonphysical alone vs baseline | **−0.151** |

Stacking every family underperforms `baseline+rate`. Extra dimensions overfit
the two non-milling videos.

## Simulation realism

Original engine `src/sim/model_fast.py` + `configs/behaviors/*.yaml` is
untouched. New opt-in engine: `src/sim/realistic.py` (speed-dependent
ω_max, OU heading/speed noise, optional `init_from`). Fitted YAMLs:
`configs/behaviors_fit/` (not used by `generate_sims.py`).

Comparison on **10-fish real annotated frames** vs 6 seeds × 150 sim frames,
KS two-sample and Wasserstein / σ_real. Raw mean Wasserstein is dominated by
NND and R_gyr (tens of pixels) and is not a useful scalar.

### Failed retunes (negative results)

1. **Median window-level Tobit MLE weights** from `annotated_config.csv`
   (`w_o ≈ 0.08`) as generative YAML. Speed KS improved (0.92 → 0.29) but the
   school exploded: NND 22 → 150 px, R_gyr 41 → 372, Φ_trans 0.97 → 0.31.
   Those weights are a *local linearization*, not Couzin parameters.
2. **Radii scaled to real NND** (50 / 110 / 200) with original social weights.
   NND improved; milling collapsed (M 0.73 → 0.37) and expansion lost Φ_rad±.
   Speed and interaction scale are not independently identifiable.

### Shipped fit

Keep original Couzin **weights and radii**. Set `s_0` / `σ_s` to real-window
medians, `fps = 29.97`, `speed_dep_turn`, OU ρ = 0.2.

| Metric (mean KS over 5 labels) | original YAML | kinematic fit |
| --- | ---: | ---: |
| **mean speed** | 0.858 | **0.439** |
| NND | 0.754 | 0.699 |
| Φ_trans | 0.435 | 0.449 |
| M | 0.261 | 0.326 |
| Φ_rad± | 0.143 | 0.299 |
| aspect | 0.332 | 0.329 |
| R_gyr | 0.788 | 0.770 |
| heading entropy | 0.370 | 0.372 |
| dlogR | 0.328 | 0.264 |
| **mean KS (all)** | **0.474** | **0.439** |
| mean W / σ_real | 0.967 | 1.290 |

Traveling speed 1.21 → **2.34** px/frame (real 2.60). Traveling Φ_trans
0.968 → **0.821** (real 0.888) and heading entropy 0.238 → **0.399** (real
0.399). Expansion speed 0.85 → 1.76 (real 1.97).

**Still broken, by design of this fit:** traveling NND 23 vs real 55;
compaction NND 15 vs 47; expansion Φ_rad± overshoots (0.70 vs 0.37); milling M
drops 0.72 → 0.38 once OU noise and higher `s_0` are on. Snapshots
(`sim_real_montage.png`) still show a tight central blob versus a tank-scale
school. Side-by-side clips: `research/figures/{real,sim_old,sim_fit}_{traveling,milling,expansion}.mp4`.

Histograms: `research/figures/sim_real_histograms.png`. Per-cell table:
`research/results/sim_realism_table.csv`.

Sim-to-real pretraining was not run: the remaining NND / R_gyr mismatch is
large enough that extra sim windows would teach the wrong scale.

## Iteration 2 — all group sizes (segment-level LOVO)

`scripts/research_scaleup.py` → `research/results/scaleup_lovo.json`.
Figure: `research/figures/scaleup_stable5.png`.

Official stable counts in the joined table: 10-fish **65**, 30-fish **117**,
70-fish **106**, 150-fish **33** (one video, **LOVO impossible**). Joined
row totals including transients: 103 / 210 / 200 / 56.

The 15-frame window protocol **cannot be repeated** at 30/70/150 without the
H5s. What replicates is the **segment-level** `dlogR` add-on. The full rate
bundle (`dlogR` + Φ slopes + `vrad_norm`) is best **pooled** (0.930) but is
not uniformly best per N (30-fish +dlogR 0.859 > +rate 0.853; 70-fish
0.912 > 0.901).

### Stable 5-class macro-F1 (segment-level LOVO)

| Split | n stable | videos | 4 Φ | **+dlogR** | +rate | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pooled (10 videos)** | 321 | 10 | 0.865 | **0.898** | **0.930** | 0.919 |
| 10-fish | 65 | 3 | 0.661 | **0.702** | 0.678 | 0.634 |
| 30-fish | 117 | 3 | 0.831 | **0.859** | 0.853 | 0.852 |
| 70-fish | 106 | 3 | 0.870 | **0.912** | 0.901 | 0.857 |
| 150-fish | 33 | 1 | — | — | — | skipped |

Pooled per-class F1 (4 Φ → +rate): traveling 0.92→0.95, milling 0.92→0.97,
shoaling 0.90→0.95, expansion **0.74→0.86**, compaction 0.84→0.91.

**`dlogR` replicates at 30- and 70-fish.** Expansion F1 at 30-fish 0.77→0.81;
at 70-fish 0.81→0.89. Milling is usable once more than one video contains it
(30-fish milling F1 0.88; 70-fish 0.87→0.93). The 10-fish milling F1 remains 0
under LOVO.

These numbers are **much higher than 5-frame windows** because a whole segment
is one sample: Φ and dlogR are averaged over seconds, not 0.17 s. They answer
“does dlogR still help when N grows?”, not “replace the 5-frame classifier
with 0.93”.

Pooled coarse-6 **transient F1**: 4 Φ 0.651 → +dlogR 0.772 → +rate **0.835**.
Same caveat: segment-level.

Permutation (pooled +dlogR): Φ_trans 0.223, Φ_tan 0.160, Φ_rad± 0.118,
Φ_local 0.083, **dlogR 0.033**. The F1 gain is real; dlogR is partly collinear
with Φ_rad± at segment scale (less so at 5-frame scale, where it ranked 2nd).

**Graph-spectral / other non-physical families at large N:** not measured.
The H5s were not on disk. Do not generalize the N=10 spectral negative.

## Iteration 2 — transients as sequences (10-fish)

`scripts/research_sequences.py` → `research/results/sequences.json`.
Tolerance ±45 frames (~1.5 s). 38 official A→B events across 3 videos.

| Detector | event P | event R | event F1 | A→B identity among TP | stable-window macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| i.i.d. 5-frame XGB (windows as events) | 0.126 | 0.816 | **0.218** | 0.194 | 0.396 (stable subset of finegrained) |
| Supervised 5-state Gaussian HMM | 0.080 | 0.868 | 0.146 | 0.242 | 0.440 |
| CUSUM + nearest centroid | 0.146 | 0.342 | 0.205 | 0.308 | 0.397 |

HMM **over-fires** (415 predicted changes vs 38 true). CUSUM is more precise
but misses most morphs. Neither beats turning i.i.d. window predictions into
change events. Sequence models are **not** paper-ready on 10-fish alone.

## Iteration 2 — joint CMA-ES calibration

`scripts/research_cma_fit.py` (popsize 6, 8 iterations, 2 seeds/eval, 11
logit-constrained parameters including radii, `s_0`, `w_o`, `w_tan`, `w_rad`,
noise). Configs: `configs/behaviors_cma/` (does not overwrite `behaviors_fit/`).
Montage: `research/figures/sim_cma_montage.png`. Mean KS on a 6-seed holdout:

| | original YAML | speed-matched fit | **CMA-ES** |
| --- | ---: | ---: | ---: |
| mean KS | **0.474** | **0.439** | 0.509 |
| NND KS | 0.754 | 0.699 | **0.685** |
| R_gyr KS | 0.788 | 0.770 | **0.681** |
| speed KS | 0.858 | **0.439** | 0.879 |
| traveling Φ_trans KS | 0.549 | 0.181 | **0.136** |
| milling M (real 0.73) | 0.72 | 0.38 | 0.89 |
| compaction NND (real 47) | 18 | 15 | **8.5 (collapsed)** |

CMA moved traveling NND 23→34 (real 55) and R_gyr 42→71 (real 104) and kept a
mill, but **did not beat** the simple speed match on overall KS, and
compaction imploded. An 8-iteration 2-seed CMA is underpowered for this
11-D landscape. The scientific point stands: **speed and interaction scale
are not independently identifiable**; sequential MLE or radius scaling fails,
and a short CMA run is not a substitute for a long, constraint-heavy search.

Default `configs/behaviors_fit/` remains the speed-matched YAMLs (better
overall KS). CMA dumps and stills stay in `configs/behaviors_cma/` and
`research/figures/sim_cma_*`. Speed-match stills remain `sim_fit_*` /
`sim_real_montage.png`.

## Annotation notes (not edited)

- README still says compaction has no real labels; the current CSVs have 54
  compaction segments.
- Gaps (n = 195, median 4.5 s) sit on many transition boundaries. They may be
  intentional (uncertain morph) or incomplete. Report only; do not fill.
- Rare transients (`milling_to_compaction` n = 1, `expansion_to_compaction`
  n = 2) cannot support a 20-way classifier.
- `annotated_features.csv` expansion/compaction n = 5 / 3; any segment-level
  metric on those two classes is noise.

## Iteration 3 — 5-frame and 15-frame LOVO on all H5s

`scripts/fetch_trajectories.py` then `scripts/research_largen.py`.
JSON: `research/results/largen_lovo_w{5,15}_h5.json`.
Figures: `research/figures/largen_stable5_w{5,15}_h5.png`.
Spectral family now also has Delaunay λ₂/gap, 0-homology component counts at
50/100/200 px, and temporal |Δλ₂| across the window.

150-fish skipped for per-N LOVO (1 video). Milling F1 is 0 at 10-fish (one
video) and **0.88–0.98** once 30/70-fish videos are in the split.

### Stable 5-class macro-F1, w=5 hop=5

| Split | n stable | 4 Φ | **+rate** | +spectral | physical | nonphys | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pooled** | 35 654 | 0.715 | **0.748** | 0.746 | **0.801** | 0.539 | 0.795 |
| 10-fish | 12 635 | 0.471 | **0.519** | 0.444 | 0.507 | 0.318 | 0.493 |
| 30-fish | 11 295 | 0.697 | **0.718** | 0.694 | 0.721 | 0.491 | 0.723 |
| 70-fish | 9 201 | 0.779 | **0.812** | 0.785 | 0.800 | 0.566 | 0.808 |
| 150-fish | 2 939 | — | — | — | skipped | — | — |

Pooled expansion F1 0.375 → 0.450 (+rate) → **0.592** (physical).
`dlogR` permutation rank: 3rd in physical (ΔF1 0.075), after Φ_trans and M.

**`dlogR` / rate replicates at 5-frame windows** at 10, 30, and 70 fish.
10-fish matches iteration 1 (0.471 → 0.519).

**Graph-spectral at large N.** `baseline+spectral` **hurts** at 10-fish
(0.471→0.444) and 30-fish (0.697→0.694), **helps slightly** at 70-fish
(0.779→0.785) and pooled (0.715→0.746). Non-physical **alone** never beats
4 Φ (pooled 0.539 vs 0.715). Stacking spectral onto physical (`all`) is
slightly worse than physical at w=5 pooled (0.795 vs 0.801). Compaction F1
is where spectral shows up pooled (0.421 → 0.512). Non-physical permutation
is dominated by heading entropy and correlation length, not λ₂.

### Stable 5-class macro-F1, w=15 hop=5

| Split | n stable | 4 Φ | **+rate** | +spectral | physical | nonphys | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pooled** | 34 968 | 0.734 | 0.767 | 0.771 | **0.820** | 0.548 | 0.814 |
| 10-fish | 12 497 | 0.524 | **0.575** | 0.488 | 0.539 | 0.333 | 0.544 |
| 30-fish | 11 053 | 0.733 | **0.761** | 0.716 | 0.743 | 0.486 | 0.743 |
| 70-fish | 8 971 | 0.794 | 0.821 | 0.815 | 0.833 | 0.577 | **0.843** |

10-fish w=15 **replicates iteration 1 on macro-F1** (4 Φ 0.524, +rate 0.575;
expansion 0.457→0.607 vs 0.618 on the earlier 10-fish-only cache). At
**70-fish**, physical+nonphysical (`all`) is the
best cell (**0.843**), and `baseline+spectral` beats 4 Φ (0.815 vs 0.794).
That is the first split where graph features carry structure. They still
do not replace Φ; they are a small add-on at N ≈ 70, w=15.

Pooled expansion F1 0.405 → 0.490 (+rate) → **0.633** (physical).

### Transients (window-level)

| Split | w | coarse-6 transient F1 (4 Φ → +rate → physical → all) | binary transient F1 (4 Φ → physical) |
| --- | --- | --- | --- |
| Pooled | 5 | 0.360 → 0.418 → 0.482 → 0.487 | 0.472 → 0.511 |
| 10-fish | 5 | 0.289 → 0.317 → 0.364 → 0.366 | 0.311 → 0.309 |
| 30-fish | 5 | 0.408 → 0.438 → 0.435 → 0.453 | 0.470 → 0.446 |
| 70-fish | 5 | 0.564 → 0.614 → 0.604 → 0.600 | 0.614 → 0.623 |
| Pooled | 15 | 0.402 → 0.475 → 0.516 → **0.524** | — |
| 70-fish | 15 | 0.602 → **0.650** → 0.639 → 0.637 | — |

Fine-grained A→B window macro-F1 stays ~0.14–0.22. Rate/physical help the
**transient superclass**, not 20-way identity.

### Sequences on all 10 videos

`scripts/research_sequences.py` → `research/results/sequences_all.json`.
248 true events, ±45 frames.

| Detector | P | R | event F1 | A→B identity | n pred |
| --- | ---: | ---: | ---: | ---: | ---: |
| i.i.d. windows as events | 0.141 | 0.548 | 0.224 | 0.346 | 964 |
| Supervised Gaussian HMM | 0.139 | 0.710 | **0.233** | 0.392 | 1264 |
| CUSUM | 0.168 | 0.077 | 0.105 | 0.474 | 113 |

HMM now **barely** beats i.i.d. events (0.233 vs 0.224) and window-stable
macro-F1 is 0.733 (milling is no longer a LOVO zero). It still over-fires
(1264 vs 248). Not a methods paper.

## Iteration 3 — longer constrained CMA-ES

`scripts/research_cma_fit.py --maxiter 20 --popsize 12 --seeds 3 --init-from-fit`
against **10-fish only** real pools (mixing 30/70/150 into the KS target was
a bug in an aborted run; the numbers below are 10-fish). Configs:
`configs/behaviors_cma_long/`. Montage:
`research/figures/sim_cma_long_montage.png`.

| | original | speed-match | short CMA | **long CMA** |
| --- | ---: | ---: | ---: | ---: |
| mean KS | 0.474 | 0.439 | 0.509 | **0.386** |
| speed KS | 0.858 | **0.439** | 0.879 | **0.413** |
| NND KS | 0.754 | 0.699 | 0.685 | **0.451** |
| R_gyr KS | 0.788 | 0.770 | 0.681 | **0.509** |
| traveling NND (real 55) | 23 | 23 | 34 | **48** |
| traveling R_gyr (real 104) | 42 | 128 | 71 | **81** |
| traveling speed (real 2.60) | 1.21 | 2.34 | 0.96 | **2.46** |
| milling M (real 0.73) | 0.72 | 0.38 | 0.89 | **0.77** |
| compaction NND (real 47) | 18 | 15 | 8.5 | **38** |

Milling held. Compaction did **not** implode. Expansion R_gyr overshot
(209 vs real 99). Init from the speed-matched YAML + collapse penalties
(`nnd < 0.4 × real`, compaction `nnd < 22`) is what the short run lacked.

Sim-to-real (5-frame 4 Φ, 8 seeds of N=10 clips, XGB trained on sim /
tested on real; **not** the old 14 k-window generate_sims protocol):

| Engine | macro-F1 on 10-fish real | macro-F1 on all-N real |
| --- | ---: | ---: |
| original YAML | 0.240 | 0.217 |
| speed-match | 0.256 | 0.341 |
| **long CMA** | **0.349** | **0.412** |

Closer kinematics transfer better in this matched protocol. Absolute F1 is
still well below real-data LOVO (0.80) because 720–1200 sim windows and a
4 Φ feature set are thin.

## Iteration 4 — operator-theoretic features (unscaled kNN)

Dedicated write-up: [`research/OPERATOR_REPORT.md`](OPERATOR_REPORT.md).
Extractor: `src/features/operator.py`. Eval: `scripts/research_operator.py`.
Delaunay is gone. Distances and velocities are **not** divided by median NND.
\(k\) was swept at segment-level grouped LOVO then frozen at **\(k=8\)**.

Pooled stable5 macro-F1, same windows/splits as iteration 3:

| Scale | n | 4 Φ | physical | Hodge | **operator** | op+phys |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| segment (591 clips) | 343 | 0.929 | **0.960** | 0.644 | 0.823 | 0.952 |
| w=15 | 34 968 | 0.734 | **0.820** | 0.538 | 0.655 | 0.818 |
| w=5 | 35 654 | 0.715 | **0.801** | 0.536 | 0.619 | 0.801 |

Operator does **not** beat 4 Φ. Signed Hodge radial flux + `dyn_dlog_ell`
are the features that matter (permutation). \(k\)-sweep pooled: 0.753 (k=3)
→ **0.823 (k=8)** → 0.820 (k=9). Nested LOVO preferred k=9 in 6/10 videos.

## Iteration 5 — optimize operators (still no Φ)

Log: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md). Eval:
`scripts/research_operator_opt.py`. Segment grouped LOVO froze **k=9** and
Hankel delays=3 rank=4 on v4 + connection + Helmholtz–Hodge 2-complex (79
features). Kernel-EDMD RBF + VAMP-2 were in the sweep; VAMPNet was not
(`torch` absent).

Pooled stable5: frozen opt **0.836 / 0.686 / 0.667** vs v4 0.820 / 0.659 /
0.620 vs 4 Φ 0.929 / 0.734 / 0.715 vs physical 0.960 / 0.820 / 0.801
(segment / w=15 / w=5). The window lift is `v4+hodge2` (0.689 / 0.658).
Connection ~0.34 alone. KEDMD 0.174 at w=5. Kitchen-sink 115 cols lose at
segment (0.815). SHAP/perm: `dyn_dlog_ell`, `h2_grad_frac`, `hodge_rad_mean`.
min-dwell Viterbi event F1 0.168 (w=15) / 0.211 (w=5), same as unconstrained
HMM, still over-fires. 10-fish is worse than v4 under the pooled freeze.

## Reproduce

```bash
python scripts/fetch_trajectories.py          # H5s into gitignored schooling-datasets/
python scripts/research_audit.py
python scripts/research_features.py --window 5 --hop 5
python scripts/research_features.py --window 15 --hop 5
python scripts/research_largen.py --window 5 --hop 5
python scripts/research_largen.py --window 15 --hop 5
python scripts/research_scaleup.py
python scripts/research_sequences.py
python scripts/research_sim_realism.py --seeds 6 --n 10
python scripts/research_cma_fit.py --maxiter 20 --popsize 12 --seeds 3 --init-from-fit --config-dir configs/behaviors_cma_long --tag cma_long
python scripts/research_sim2real.py
python scripts/research_plots.py
python scripts/research_operator.py --scale ksweep
python scripts/research_operator.py --scale all --skip-ksweep
python scripts/research_operator_opt.py --mode sweep
python scripts/research_operator_opt.py --mode lovo
```

Original training / eval CLIs are unchanged. Speed-matched YAMLs:
`configs/behaviors_fit/`. Constrained long CMA: `configs/behaviors_cma_long/`.

## Verdict — what is paper-worthy

1. **Yes: `dlogR` as a physical coordinate.** Replicated at 5-frame and
   15-frame windows, at 10/30/70 fish, and in the earlier segment-level
   table. Expansion F1 is the class that moves. Phase portrait
   (Φ_trans, Φ_rad±, dlogR) is the figure to write.
2. **Yes, with a caveat: graph-spectral add-ons at N ≈ 70.** They hurt at
   N=10, are ~neutral at N=30, and help at 70-fish w=15 (`all` 0.843 vs
   physical 0.833 vs 4 Φ 0.794). Non-physical **alone** is not a classifier.
   This is a short result, not a TDA paper.
3. **Yes, as methods: constrained joint Couzin calibration.** Long CMA
   (pop 12, 20 iter, init from speed-match, collapse penalties) is the first
   generator that moves NND and R_gyr **without** killing the mill or
   imploding compaction (mean KS 0.474 → **0.386**, traveling NND 23 → 48).
   Sim-to-real improves 0.240 → 0.349 on 10-fish in a small matched test.
   Expansion scale is still wrong. Do not claim “the sim looks like the
   videos” from the montage alone.
4. **Not yet: transients as sequences.** All-N HMM event F1 0.233 vs i.i.d.
   0.224, 1264 predicted vs 248 true. Superclass F1 0.36 → 0.52 is real but
   A→B identity is not. Needs an HSMM / duration model, not another XGB.
5. **Not a Φ replacement: unscaled operator features.** Full write-up
   `research/OPERATOR_REPORT.md`. Domain-agnostic kNN + signed Hodge + DMD
   reaches 0.823 / 0.655 / 0.619 (segment / w=15 / w=5) vs 4 Φ 0.929 /
   0.734 / 0.715. The useful coordinates are `hodge_rad_*` and
   `dyn_dlog_ell` — the operator twins of \(\Phi_{\mathrm{rad}}\) and
   dlogR — not Laplacian \(\lambda_2\) and not short-window Koopman.
   Methods note: **do not NND-scale** if expansion/compaction matter.
6. **Not a Φ replacement after optimizing operators.** Frozen k=9 + Hodge
   2-complex + Hankel-d3-r4 reaches 0.836 / 0.686 / 0.667. Helmholtz–Hodge
   on kNN triangles is the only new family that moves window F1. SO(2)
   connection and Kernel-EDMD do not. Chronological log:
   [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
