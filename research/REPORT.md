# Feature space and simulation realism

Josef Berman, Swarm & AI Lab (University of Haifa). Research branch
`research/feature-space-and-sim`. Annotation files were not modified.

Every number below comes from a script in this branch. JSON dumps live in
`research/results/`. Figures live in `research/figures/`.

## Questions

1. With a **leakage-safe** protocol (leave-one-video-out), how well do the four
   published order parameters classify the five stable CouzinLab behaviours and
   the transients between them?
2. Which **new feature families** — physical or not — move macro-F1, especially
   on expansion/compaction and on transients?
3. Why do YAML simulations **not look like** the real 10-fish videos, and can a
   kinematic retune close the gap without destroying the order-parameter
   signatures that the YAMLs already produce?

## Data and labels (audit)

Source: `annotations/*.csv` (10 videos, **591 segments**). Transients are
canonical labels containing `_to_`. No unknown labels. **No overlapping
segments.** **195 gaps** between consecutive annotations (median **4.50 s**);
those frames are unlabeled, not relabeled.

| Kind | Segments | Dominant classes |
| --- | ---: | --- |
| Stable | 343 | traveling 165, compaction 54, shoaling 51, expansion 38, milling 35 |
| Transient | 248 | traveling_to_compaction 38, traveling_to_shoaling 36, shoaling_to_traveling 34, expansion_to_traveling 30, traveling_to_milling / milling_to_traveling 21 each; 11 other `A_to_B` labels with n ≤ 14 |

**Trajectory coverage in this clone.** CouzinLab H5 files are gitignored.
Only the three **10-fish** videos (`0066`, `0105`, `0126`) were loadable
(Drive download cap). That is **107 / 591** annotated segments,
**13 277** non-overlapping 5-frame windows. All **188 milling windows** are in
`0066` only, so leave-one-video-out milling F1 is structurally 0. The 30 / 70 /
150-fish H5 files were not obtained.

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

## Annotation notes (not edited)

- README still says compaction has no real labels; the current CSVs have 54
  compaction segments.
- Gaps (n = 195, median 4.5 s) sit on many transition boundaries. They may be
  intentional (uncertain morph) or incomplete. Report only; do not fill.
- Rare transients (`milling_to_compaction` n = 1, `expansion_to_compaction`
  n = 2) cannot support a 20-way classifier.
- `annotated_features.csv` expansion/compaction n = 5 / 3; any segment-level
  metric on those two classes is noise.

## Reproduce

```bash
python scripts/research_audit.py
python scripts/research_features.py --window 5 --hop 5
python scripts/research_features.py --window 15 --hop 5
python scripts/research_sim_realism.py --seeds 6 --n 10
python scripts/research_plots.py
```

Original training / eval CLIs are unchanged. Fitted YAMLs are only read by
`src.sim.realistic.run_realistic`.

## Recommended next steps (paper)

1. **`dlogR` as the missing physical coordinate for E+ / E−.** Pair Φ_trans,
   Φ_rad±, and d log R / dt in a 3-D phase portrait on all group sizes (need
   the 30/70/150-fish H5). That is a small, non-trivial physical finding and
   the only family that moved expansion F1 by a large margin.
2. **Transients as change-points, not extra classes.** Binary / 20-way window
   classification failed. A hidden Markov or CUSUM detector on
   (Φ_trans, dlogR), trained leave-one-video-out, matches how the labels were
   drawn (morphs between stables) and uses the 248 transient segments as
   event times rather than i.i.d. windows.
3. **Joint Couzin calibration, not sequential.** Window MLE weights are not
   generative; scaling radii independently kills milling. Fit
   `(s_0, r_r, w_o, w_tan, w_rad, σ_θ)` by CMA-ES on the vector of KS
   statistics (speed, NND, Φ_trans, M, Φ_rad±, R_gyr). Then test whether the
   resulting clips help real-data training (sim-to-real) — they will not,
   until NND and R_gyr match.

Secondary, after the 30-fish H5 is local: re-run spectral / TDA families. At
N = 10 the interaction graph is too small for λ₂ to beat Φ_trans; that
negative result should not be generalized to 70–150 fish.
