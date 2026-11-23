# Operator-theoretic features (unscaled kNN)

Josef Berman, Swarm & AI Lab (University of Haifa). Branch
`research/feature-space-and-sim`. Annotation files were not modified.
H5 trajectories from `https://huggingface.co/buckets/josefberman/SchoolingFish`.
Every number below is from `scripts/research_operator.py` and
`scripts/research_operator_opt.py` (JSON in `research/results/operator_*.json`
and `operator_opt_*.json`). Chronological run log:
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

Linked from `research/REPORT.md`. This pass **drops all hand-crafted Φ / M /
dlogR** from the feature vector and asks whether a purely operator-theoretic
representation of the collective trajectories can match the physical baseline
under the same grouped leave-one-video-out protocol, at three temporal scales.

## Design

Permutation-invariant across fish (Laplacian spectra, Hodge 1-form scalars,
graph-Fourier DMD). Translation-invariant (COM-centered positions). Rotation
invariant as scalars. **Not scale-normalized:** distances and velocities stay
in native units. Per-frame median-NND scaling was tried first and **erased**
expansion/compaction; Josef’s instruction was to keep the scale.

No Delaunay. The interaction graph is a symmetrized unscaled \(k\)-NN graph
with inverse-distance weights \(w_{ij} = 1/d_{ij}\) on the \(k\)-NN support.
\(k\) is capped at \(N-1\) (so 10-fish with \(k=9\) is a 9-NN graph).

### Families (`src/features/operator.py`, 61 features)

**Spatial.** Normalized Laplacian \(L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}\):
Fiedler \(\lambda_2\), gap, \(\lambda_{\max}\), \(\lambda_{3\ldots 6}\),
moments \(\mathrm{mean}(\lambda^p)\), spectral entropy, heat-kernel traces
\(\mathrm{mean}(e^{-t\lambda})\) at \(t\in\{0.25,1,4\}\), component count.
Combinatorial \(\lambda_2(D-W)\) (scale-sensitive). Mean/median kNN edge
length \(\bar\ell\).

**Magnetic.** Hermitian magnetic Laplacian \(L^{(q)}\) with phase
\(\theta_{ij} = \mathrm{atan2}(e_{ij}\times v_{\mathrm{avg}}, |e_{ij}|\,|v|)\)
at charges \(q\in\{0.25,0.5\}\). Real spectrum + \(\Delta\lambda_2 = \lambda_2^{(q)}-\lambda_2\).

**Hodge (signed 1-form).** On kNN edges, \(\alpha_{ij} = v_{\mathrm{avg}}\cdot\hat e_{ij}\)
(signed stretch), \(\rho_{ij} = v_{\mathrm{avg}}\cdot\hat r_{ij}\) (signed radial
flux about COM), \(\omega_{ij}\) (2-D curl). Features: \(\mathrm{mean}(\alpha)\)
and \(\mathrm{std}(\alpha)\), \(\mathrm{mean}(\rho)\) and \(\mathrm{std}(\rho)\),
unsigned energies \(\mathrm{mean}(\alpha^2)\), \(\mathrm{mean}(\omega^2)\),
curl fraction. **Sign is the point:** expansion vs compaction.

**Dyn.** \(\Delta\lambda_2\), spectral \(W_1\), Fiedler correlation, kNN Jaccard,
and unscaled edge-length growth \(\Delta\bar\ell/\Delta t\) and
\(\Delta\log\bar\ell/\Delta t\).

**DMD / Hankel-DMD.** Exact DMD on graph-Fourier coefficients of **raw**
velocity (first \(R=4\) nontrivial eigenvectors of \(L_{\mathrm{sym}}\) at
window midpoint). Growth \(\mathrm{Re}(\log\mu)/\Delta t\), frequency
\(|\mathrm{Im}(\log\mu)|/\Delta t\) (Nyquist-pinned modes dropped), spectral
radius, residual, mode concentration.

**Koopman / HAVOK.** Per-step residual \(\|\psi(x_{t+1})-A\psi(x_t)\|\) and
HAVOK forcing statistics of subsampled \(\lambda_2(t)\).

### \(k\)-sweep (frozen before the three-scale run)

Candidates \(k\in\{3,4,5,6,7,8,9\}\). Protocol: **segment-level grouped LOVO**
on the full 61-D operator suite (XGB), 591 H5 segments, 10 videos. Selection
criterion: argmax pooled stable-5 macro-F1. Nested LOVO (pick \(k\) on the
other 9 videos, score the held-out video) is reported as a robustness check,
not as the freeze rule.

Figure: `research/figures/operator_k_sweep.png`.
JSON: `research/results/operator_k_sweep.json`.

| \(k\) | pooled stable5 | trans. F1 | 10-fish | 30-fish | 70-fish |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.753 | 0.719 | 0.625 | 0.595 | 0.664 |
| 4 | 0.757 | 0.722 | 0.636 | 0.610 | 0.666 |
| 5 | 0.807 | 0.726 | 0.597 | 0.589 | **0.718** |
| 6 | 0.806 | 0.730 | 0.624 | 0.633 | 0.631 |
| 7 | 0.804 | 0.736 | 0.633 | 0.637 | 0.659 |
| **8** | **0.823** | 0.740 | 0.624 | 0.660 | 0.643 |
| 9 | 0.820 | **0.740** | **0.733** | **0.676** | 0.677 |

**Frozen \(k^\star = 8\).** Nested inner-LOVO chose \(k=9\) in 6/10 videos,
\(k=8\) in 3, \(k=5\) in 1 (video 0066). 10-fish alone prefers \(k=9\)
(0.733 vs 0.624). Pooled \(k=8\) vs \(k=9\) is a 0.004 gap; we freeze the
declared pooled argmax and do not retune \(k\) on window data.

## Protocol (apples-to-apples)

Same 10 videos, same official labels, same grouped LOVO
(`src/classify/grouped.py`), class-weighted XGB unless noted. Window rows
are **exactly** the largen-cache windows (w=15: 39 304; w=5: 40 486; hop=5).
Segment rows are the 591 official H5 clips. 150-fish is one video (N=151):
included in pooled, skipped per-N.

Tasks: stable5, coarse-6 (transient superclass), binary transient,
fine-grained A→B. Ablations: spatial / magnetic / Hodge / DMD / Koopman /
dyn / combined operator vs 4 Φ vs full physical vs operator+physical.
Classifiers on the operator suite: XGB, \(\ell_2\)-logistic, HistGradientBoosting.
Permutation importance: 3 shuffles, pooled stable5, XGB only.

Physical 4 Φ / full physical numbers at w=5 and w=15 **match**
`largen_lovo_w{5,15}_h5.json` (0.715 / 0.801 and 0.734 / 0.820 pooled
stable5). Segment 4 Φ / physical are recomputed on H5s (not the iteration-2
`annotation_review` join).

## Stable 5-class macro-F1

### Pooled (10 videos)

| Scale | n stable | 4 Φ | physical | spatial | magnetic | Hodge | DMD | Koopman | dyn | **operator** | op+phys |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **segment** | 343 | 0.929 | **0.960** | 0.288 | 0.407 | 0.644 | 0.495 | 0.295 | 0.529 | 0.823 | 0.952 |
| **w=15** | 34 968 | 0.734 | **0.820** | 0.287 | 0.403 | 0.538 | 0.311 | 0.306 | 0.340 | 0.655 | 0.818 |
| **w=5** | 35 654 | 0.715 | **0.801** | 0.290 | 0.402 | 0.536 | 0.234 | 0.213 | 0.322 | 0.619 | 0.801 |

### Per-N (operator vs 4 Φ vs physical)

| Scale | split | n stable | 4 Φ | physical | operator | op+phys |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| segment | 10-fish | 69 | **0.744** | 0.687 | 0.624 | 0.687 |
| segment | 30-fish | 121 | 0.886 | 0.908 | 0.660 | **0.911** |
| segment | 70-fish | 115 | **0.905** | 0.886 | 0.643 | 0.886 |
| w=15 | 10-fish | 12 497 | 0.524 | 0.539 | 0.397 | **0.540** |
| w=15 | 30-fish | 11 053 | 0.733 | 0.743 | 0.536 | **0.746** |
| w=15 | 70-fish | 8 971 | 0.794 | 0.833 | 0.624 | **0.835** |
| w=5 | 10-fish | 12 635 | 0.471 | **0.507** | 0.351 | 0.487 |
| w=5 | 30-fish | 11 295 | 0.697 | **0.721** | 0.470 | 0.718 |
| w=5 | 70-fish | 9 201 | 0.779 | **0.800** | 0.589 | 0.799 |

Pooled per-class F1 (operator → 4 Φ → physical):

| Scale | T | M | S | E+ | E− |
| --- | ---: | ---: | ---: | ---: | ---: |
| segment op | 0.917 | 0.714 | 0.808 | 0.815 | 0.862 |
| segment 4 Φ | 0.960 | 0.943 | 0.961 | 0.857 | 0.926 |
| segment physical | 0.982 | 0.941 | 0.981 | 0.961 | 0.936 |
| w=15 op | 0.930 | 0.827 | 0.705 | 0.373 | 0.440 |
| w=15 4 Φ | 0.934 | 0.967 | 0.910 | 0.405 | 0.452 |
| w=15 physical | 0.971 | 0.975 | 0.936 | 0.633 | 0.586 |
| w=5 op | 0.914 | 0.778 | 0.686 | 0.317 | 0.397 |
| w=5 4 Φ | 0.919 | 0.959 | 0.900 | 0.375 | 0.421 |
| w=5 physical | 0.961 | 0.972 | 0.932 | 0.592 | 0.548 |

Operator is within ~0.03 of 4 Φ on traveling at every scale. The deficit is
**milling at N=10** (LOVO zero whenever milling is in one video) and
**expansion/compaction at window scale**, where 4 Φ already struggles and
physical `dlogR` is the known fix. At segment scale, operator expansion
0.815 is close to 4 Φ 0.857 — the signed Hodge radial flux and
\(\Delta\log\bar\ell\) are doing that job.

## Transients

### Coarse-6 transient-class F1

| Scale | split | 4 Φ | physical | Hodge | DMD | **operator** | op+phys |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| segment | pooled | 0.657 | 0.852 | 0.500 | 0.680 | **0.740** | **0.865** |
| segment | 10-fish | 0.697 | 0.789 | 0.462 | 0.588 | 0.744 | 0.778 |
| segment | 30-fish | 0.714 | 0.809 | 0.574 | 0.660 | 0.737 | 0.813 |
| segment | 70-fish | 0.671 | 0.811 | 0.524 | 0.688 | 0.740 | **0.896** |
| w=15 | pooled | 0.402 | 0.516 | 0.223 | 0.092 | 0.290 | 0.531 |
| w=5 | pooled | 0.360 | 0.482 | 0.220 | 0.116 | 0.286 | 0.497 |

### Binary transient / fine-grained A→B (pooled)

| Scale | binary op | binary 4 Φ | binary physical | fine op | fine 4 Φ | fine physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| segment | 0.722 | 0.731 | **0.847** | 0.268 | 0.334 | **0.502** |
| w=15 | 0.606 | 0.687 | **0.718** | 0.157 | 0.214 | **0.246** |
| w=5 | 0.599 | 0.672 | **0.704** | 0.149 | 0.195 | **0.220** |

Operator **beats 4 Φ on the segment-level transient superclass** (0.740 vs
0.657) and is slightly under on binary. At 5/15-frame windows it does not.

### A→B events (±45 frames, 248 true)

Window-level HMM / i.i.d. XGB / Koopman-residual CUSUM on operator features.
Not a win.

| Detector | w=15 event F1 | w=5 event F1 | n pred (w=15) |
| --- | ---: | ---: | ---: |
| operator HMM | 0.180 | 0.204 | 630 |
| i.i.d. operator windows | 0.130 | 0.128 | 2581 |
| residual CUSUM | 0.117 | 0.000 | 301 |

Physical-feature HMM from iteration 3 was 0.233. Residual CUSUM does not
pinpoint transitions.

## Classifiers (operator suite, pooled stable5)

| Scale | XGB | logreg | HGB |
| --- | ---: | ---: | ---: |
| segment | 0.823 | 0.761 | **0.825** |
| w=15 | **0.655** | 0.620 | 0.648 |
| w=5 | **0.619** | 0.608 | 0.609 |

MLP was dropped after it failed on singleton A→B labels (fine-grained) and
lost to XGB on segment (0.313 in the NND-scaled run). LightGBM / CatBoost /
torch are not installed. Trees win; the feature set is the lever, not the
head.

## What actually drives the operator suite

Permutation Δ macro-F1, pooled stable5, XGB, 3 shuffles:

| Rank | segment | w=15 | w=5 |
| ---: | --- | --- | --- |
| 1 | `dyn_dlog_ell` 0.109 | `hodge_rad_std` 0.132 | `hodge_rad_std` 0.143 |
| 2 | `hodge_rad_mean` 0.092 | `hodge_rad_mean` 0.078 | `hodge_rad_mean` 0.083 |
| 3 | `hodge_rad_std` 0.088 | `koop_resid_mean` 0.034 | `mag_q25_dl2` 0.032 |
| 4 | `mag_q25_dl2` 0.048 | `dyn_dlog_ell` 0.033 | `hodge_div_std` 0.023 |
| 5 | `dmd_radius` 0.042 | `mag_q25_dl2` 0.027 | `hodge_grad` 0.015 |

The Laplacian eigenvalue vector (spatial) is **not** the classifier. Signed
radial 1-form energy and unscaled edge-length growth are. Those are the
operator-theoretic twins of \(\Phi_{\mathrm{rad}\pm}\) and \(\mathrm{d}\log R\),
arrived at without writing those symbols down. Magnetic \(\Delta\lambda_2\)
is a real but smaller circulation term. DMD/Koopman help at **segment**
duration (DMD family 0.495; DMD transient F1 0.680) and collapse at 5/15
frames (too short for a useful spectrum).

Figures: `operator_stable5_{segment,w15_h5,w5_h5,three_scales}.png`,
`operator_planes_*.png`, `operator_residual_w{5,15}_h5_0103.png`,
`operator_transient_three_scales.png`.

## Verdict

**Not a replacement for 4 Φ or the physical suite.** Pooled stable5:
operator 0.823 / 0.655 / 0.619 vs 4 Φ 0.929 / 0.734 / 0.715 vs physical
0.960 / 0.820 / 0.801 (segment / w=15 / w=5). Stacking operator onto
physical is ~neutral (0.952 / 0.818 / 0.801).

**Yes, as a methods result.** (1) Do not NND-scale the graph if
expansion/compaction matter — the scaled extractor sat at ~0.48 segment
stable5; unscaled \(k=8\) is 0.823. (2) Freeze \(k\) with grouped LOVO;
\(k=8\) wins pooled, \(k=9\) wins nested and 10-fish. (3) The useful
coordinates in this dictionary are signed Hodge radial flux and
\(\Delta\log\bar\ell\), not \(\lambda_2\) of a static kNN Laplacian.
That is domain-agnostic reconstruction of the same physics the order
parameters already name.

**Not paper-worthy as “Koopman/DMD beats Φ.”** Short-window DMD/Koopman
families are the weakest ablation. Residual CUSUM does not find A→B
events. The operator suite is a ~89% shadow of 4 Φ at every scale
(0.823/0.929, 0.655/0.734, 0.619/0.715), which is interesting and not a
win.

Reproduce:

```bash
python scripts/research_operator.py --scale ksweep
python scripts/research_operator.py --scale all --skip-ksweep
python scripts/research_operator_opt.py --mode sweep
python scripts/research_operator_opt.py --mode lovo
```

## Iteration 5 — connection, Hodge 2-complex, Kernel-EDMD, Hankel tune

Hypothesis: SO(2) connection Laplacian (Singer 2012), Helmholtz–Hodge on kNN
triangles, RBF Kernel-EDMD + linear VAMP-2, and a Hankel delay/rank grid can
close the remaining gap to 4 Φ without writing Φ. VAMPNet / deep Koopman
skipped (`torch` not installed). Hyps selected on **segment grouped LOVO**
then frozen: **k=9**, Hankel delays=3 rank=4 on top of v4 + connection +
hodge2 (79 features). Sweep JSON: `operator_opt_sweep.json`. Near-tie:
`base+kedmd_s05_r4` 0.835 vs winner 0.836. Kitchen-sink 115 columns **lost**
(0.815).

Pooled stable5 / coarse-6 transient F1 (`operator_opt_lovo_*.json`):

| Scale | v4 (k=9) | +hodge2 | **opt (frozen)** | all 115 | 4 Φ | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| segment | 0.820 | 0.814 | **0.836** | 0.815 | 0.929 | 0.960 |
| w=15 | 0.659 | **0.689** | 0.686 | 0.692 | 0.734 | 0.820 |
| w=5 | 0.620 | 0.658 | **0.667** | 0.667 | 0.715 | 0.801 |

Transient F1: opt 0.741 / 0.355 / 0.346 vs 4 Φ 0.657 / 0.402 / 0.360 vs
physical 0.852 / 0.516 / 0.482.

Connection alone ~0.34 at every scale; v4+connection is flat or worse.
KEDMD alone 0.425 / 0.244 / 0.174 — dies at short windows. Best classifier
on the frozen set is XGB (0.836 / 0.686 / 0.667); HGB 0.812 / 0.671 / 0.650.

Segment permutation: `dyn_dlog_ell` 0.124, `hodge_rad_mean` 0.079,
`h2_grad_frac` 0.059. SHAP agrees, with `h2_grad_frac` second. The new
coordinate that moves windows is Helmholtz gradient energy, not Kernel-EDMD.

min-dwell Viterbi (D=3) vs unconstrained HMM vs i.i.d. windows, ±45 frames:
w=15 F1 0.168 / 0.168 / 0.172; w=5 0.211 / 0.211 / 0.189. Still hundreds of
false positives. Duration constraint does not help.

10-fish is **worse** under the pooled freeze than v4 (segment 0.708 vs
0.733; w=15 0.376 vs 0.390; w=5 0.351 vs 0.377). 30/70-fish windows gain.

**Verdict, iteration 5.** Not a Φ replacement. The only operator upgrade
that moves window F1 is the kNN-triangle Helmholtz–Hodge split. Connection
and Kernel-EDMD do not. Do not dump every Koopman variant into the vector.
