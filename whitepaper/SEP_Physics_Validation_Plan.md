# SEP Engine Physics Validation Plan

**Date:** August 24, 2025
**Version:** 1.0
**Author:** Alexander Nagy (SEP Dynamics)

## Purpose

Validate or falsify the SEP framework against objective physics across scales using a minimal, high‑signal battery of experiments. Each test targets a crisp hypothesis, defines pre‑registered metrics and pass/fail thresholds, and produces artifacts suitable for peer review.

---

## Core Definitions (fixed for all tests)

**State:** $s_t \in \{0,1\}^{64}$.
**Overlap:** $O_t = \mathrm{popcount}(s_{t-1}\wedge s_t)$.
**Flips:** $F_t = \mathrm{popcount}(s_{t-1}\oplus s_t)$.
**Densities:** $n_{t-1}=\mathrm{popcount}(s_{t-1}),\; n_t=\mathrm{popcount}(s_t)$.

**Coherence (baseline‑corrected):**

$$
\mathbb{E}[O] = \frac{n_{t-1} n_t}{64},\qquad
C_t=\frac{O_t-\mathbb{E}[O]}{64-\mathbb{E}[O]}\in[0,1].
$$

**Stability:** $f_t=F_t/64$, $\;S_t=1-\mathrm{EMA}_\beta[f_t]$.

**Entropy:** $p_{i,t}=\mathrm{EMA}_\beta[s_t(i)],\;H_t=\frac{1}{64}\sum_i \big[-p_{i,t}\log p_{i,t}-(1-p_{i,t})\log(1-p_{i,t})\big]$ (bits).

**Bit mapping (pre‑registered):** from a real‑valued signal $x(t)$, build 64 bits per time step using fixed comparators (choose one and keep it fixed per dataset):

* **D1:** Sign of derivative over 64 staggered micro‑windows.
* **D2:** 64 quantile thresholds of a rolling window on $x(t)$.
* **D3:** 8 bands × 8 features (e.g., bandpass energy > rolling median).

**EMA half‑life:** choose $T_{1/2}$ per dataset a priori; $\beta=1-\exp\{-\ln 2 / T_{1/2}\}$.

---

## Hypotheses (pre‑registered)

* **H1 (Isolated invariance):** For isolated processes (no continuous reaction), triad trajectories are invariant under proper time scaling (after trivial axis rescale).
* **H2 (Verification non‑invariance):** For continuously reacting systems, naive time scaling disrupts triad trajectories unless local reaction parameters are re‑estimated.
* **H3 (Pairwise sufficiency):** Pairwise maximum‑entropy models explain most structure in bit sequences; adding third‑order interactions yields marginal improvement.
* **H4 (Convolutional invariance in waves):** For band‑limited physical waves, controlled convolution/decimation preserves triad trajectories (within error bounds).
* **H5 (Cosmological reparameterization):** After de‑redshifting light curves by $1+z$, triad trajectories align across sources (stronger than in raw time).
* **H6 (Retrodictive reconstruction):** With continuity (low flip budget), prior states can be reconstructed from triads over short windows with high accuracy on real signals.

---

## Test Battery (publishable set only)

### T1. Isolated vs Reactive Time Scaling (Particle/Bio scale)

**Targets:** H1, H2

**Datasets:**

* **Isolated:** particle decay event sequences (event times binned to regular cadence).
* **Reactive:** chemical oscillator or physiological rhythm under load (time‑series with sustained feedback).

**Method:**

1. Map each dataset to 64‑bit states using one fixed mapping (D1 or D2).
2. Compute triads over time.
3. Apply time scaling $\gamma \in \{1.2,1.5,2.0\}$ by resampling.
4. Align triad curves to baseline via time rescale (and DTW only if strictly necessary).
5. Compute RMSE between baseline triads and scaled triads (per metric and jointly).

**Pass/Fail (pre‑registered):**

* **H1 pass:** median RMSE$_\text{isolated}$ ≤ 0.05 across $\gamma$.
* **H2 pass:** median RMSE$_\text{reactive}$ ≥ 2× RMSE$_\text{isolated}$ for the same $\gamma$.
  CI by bootstrap; report effect size.

**Artifacts:** Notebook, plots (H/C/S vs time, overlaid), RMSE tables, seeds, mapping spec, half‑life used.

---

### T2. Pairwise Maximum‑Entropy Sufficiency

**Target:** H3

**Datasets:** Use bit sequences from T1 (both types), and one astrophysical time series mapped to bits (e.g., pulsar residuals).

**Method:**

1. Fit pairwise max‑entropy (Ising) model via pseudolikelihood to match first/second moments.
2. Fit extended model with selected third‑order terms (L1‑regularized).
3. Evaluate held‑out negative log‑likelihood (NLL); compute AIC/BIC deltas.

**Pass/Fail:**

* **H3 pass:** third‑order model improves held‑out NLL by < 10% over pairwise on all datasets; AIC/BIC deltas not compelling after penalty.

**Artifacts:** Code, NLL tables, AIC/BIC deltas, train/test splits, seeds.

---

### T3. Gravitational Waves: Convolutional Invariance + Retrodiction

**Targets:** H4, H6

**Dataset:** Ringdown segments of gravitational‑wave strain $h(t)$ (band‑limited, high SNR).

**Method:**

1. Map $h(t)$ to bits (D1 or D3). Compute triads.
2. **Convolutional invariance:** apply controlled smoothing/decimation (e.g., factor 2 FIR); recompute triads; align and compute RMSE.
3. **Retrodiction:** remove K intermediate states; reconstruct via continuity (flip budget $k$) and triad consistency; measure reconstruction accuracy.

**Pass/Fail:**

* **H4 pass:** median RMSE between original and decimated triads ≤ 0.05 across ringdown segments.
* **H6 pass:** reconstruction accuracy ≥ 95% for $k \le 4$ and window length $w \in [3,7]$.

**Artifacts:** Before/after triad plots, RMSE reports, reconstruction confusion matrices, code and filters used.

---

### T4. Supernovae: De‑Redshift Alignment of Triads

**Target:** H5

**Dataset:** Type‑Ia supernova light curves with measured redshifts $z$.

**Method:**

1. Interpolate each light curve to a uniform grid; map to bits (D2).
2. Compute triads in observed time; align between sources; compute alignment RMSE/DTW.
3. De‑redshift time by $1+z$; recompute triads; re‑align; recompute RMSE/DTW.
4. Compare distributions of alignment errors pre vs post de‑redshift.

**Pass/Fail:**

* **H5 pass:** median alignment error (RMSE or DTW distance) decreases by ≥ 25% after de‑redshift across the cohort (paired test, corrected p < 0.01).

**Artifacts:** Alignment plots per source, aggregated error distributions, statistical test report, mapping spec, half‑life.

---

### T5. Pulsar Timing: Pairwise Sufficiency and Triad Consistency

**Targets:** H3, H4

**Dataset:** Pulsar timing residuals (multiple pulsars).

**Method:**

1. Map residuals to bits (D1 or D3).
2. Fit pairwise vs third‑order max‑entropy models; evaluate held‑out NLL and AIC/BIC (as in T2).
3. Apply controlled smoothing/decimation to residuals; test triad invariance (as in T3).

**Pass/Fail:**

* **H3 pass:** third‑order gains < 10% across pulsars.
* **H4 pass:** triad RMSE ≤ 0.05 under decimation.

**Artifacts:** NLL tables, triad invariance plots, test statistics.

---

## Execution Order (and why)

1. **T1 (Isolated vs Reactive):** Establishes the core contrast your theory hinges on. If this fails, stop and revise.
2. **T2 (Pairwise Sufficiency):** Tests the “binary interaction suffices” claim on the same data plus one astrophysical set.
3. **T3 (LIGO):** Strong wave‑domain test of convolutional invariance and retrodiction on a clean physical signal.
4. **T4 (Supernovae):** Global‑scale time reparameterization test (de‑redshift).
5. **T5 (Pulsars):** Cross‑checks H3/H4 on another astrophysical source.

---

## Statistical Protocol (pre‑registered)

* **Alignment error:** RMSE and DTW distance on standardized triads; report per‑metric and joint (stacked vector).
* **Uncertainty:** 10k bootstrap resamples for CIs on medians/effect sizes.
* **Hypothesis tests:** Paired Wilcoxon or permutation tests; multiple‑comparison correction by Benjamini–Hochberg (q=0.05).
* **Model selection (T2/T5):** Held‑out NLL; AIC/BIC with consistent penalties across datasets.
* **Sensitivity:** Repeat analyses across bit mappings (D1/D2/D3) and EMA half‑lives; report robustness.

---

## Stopping Rules

* **Global pass:** H1–H6 all pass their thresholds on at least one mapping (D1/D2/D3) per test and remain robust under sensitivity checks.
* **Early stop:** If T1 fails (H1 fails and H2 fails), halt. If T3 fails both H4 and H6, halt and revise triad/bit mapping.
* **Revision triggers:** Any single test repeatedly fails across mappings and half‑lives with narrow CIs.

---

## Reproducibility & Artifacts

* **Per test notebook** with deterministic seeds, environment file, triad parameters, bit mapping spec.
* **Outputs:** CSVs of triads, alignment error tables, model metrics, and plots (PNG/SVG).
* **Provenance:** Commit hash, dataset version identifiers, hardware profile.
* **Packaging:** `repro_pack/` containing code, configs, and result summaries.

---

## Risks & Mitigations

* **Mapping bias:** Fix mappings D1–D3 up front; report all; avoid post‑hoc remapping.
* **Over‑smoothing:** Limit decimation (T3/T5) to factors with documented passbands; publish filters.
* **P‑hacking:** Pre‑register thresholds; keep tests minimal; correct for multiplicity.
* **Data quirks:** Use multiple sources per test where possible; include robustness analysis.

---

## Interpretation

* **Consistent passes** support the claim that the triad provides **scale‑robust, convolution‑stable observables** of time‑structured processes, and that **pairwise** interactions capture the dominant structure in physics data streams.
* **Targeted failures** identify which component (coherence normalization, stability half‑life, entropy window, or bit mapping) needs refinement.

