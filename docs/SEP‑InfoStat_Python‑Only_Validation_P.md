# SEP‑InfoStat: Python‑Only Validation Plan

**Goal**
Test whether your triad observables — **Entropy H**, **Coherence C**, **Stability S** — behave like objective, scale‑robust statistics of time‑structured processes, and whether **retrodiction** (using future data) reduces uncertainty in a measurable way. Also test your “binary interactions suffice” claim via pairwise maximum‑entropy models.

**Design principles**

* Only include tests that would interest a picky reviewer.
* Pre‑register thresholds.
* Use synthetic but physically motivated processes so results are verifiable and reproducible with seeds.
* Python only: `numpy`, `scipy`, `scikit‑learn`, `matplotlib`. Optional: `numba` for speed.

---

## 0) Environment & Reproducibility

**Environment**

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-learn matplotlib numba==0.59.1
```

**Global settings**

* Random seed: 1337 (override via CLI if you care).
* Logs in `results/` (CSV + PNG).
* Use base‑2 logs for entropy; report units clearly.

**Core triad (single source of truth)**

```python
import numpy as np

def triad(prev_bits: np.ndarray, curr_bits: np.ndarray,
          ema_flip_prev: float, ema_p_prev: np.ndarray, beta: float):
    """Compute (H,C,S) from two 64-bit states with EMAs for stability & entropy."""
    assert prev_bits.shape == (64,) and curr_bits.shape == (64,)
    # Overlap / flips
    O = np.sum(prev_bits & curr_bits)
    F = np.sum(prev_bits ^ curr_bits)
    nA, nB = prev_bits.sum(), curr_bits.sum()
    # Baseline-corrected coherence
    E_O = (nA * nB) / 64.0
    denom = max(1e-12, 64.0 - E_O)
    C = (O - E_O) / denom
    C = float(np.clip(C, 0.0, 1.0))
    # Stability via EMA of flip-rate
    f = F / 64.0
    ema_flip_t = (1 - beta) * ema_flip_prev + beta * f
    S = 1.0 - ema_flip_t
    # Entropy via EMA of bit probabilities
    ema_p_t = (1 - beta) * ema_p_prev + beta * curr_bits
    p = np.clip(ema_p_t, 1e-9, 1 - 1e-9)
    Hb = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    H = float(Hb.mean())
    return H, C, S, ema_flip_t, ema_p_t
```

**Bit mappings** (fixed choices for all tests; pick one per test and stick to it)

* **D1:** sign of derivative over 64 staggered micro‑windows.
* **D2:** 64 rolling quantile thresholds (e.g., 1.5%, 3%, …, 98.5%).
* **D3:** 8 frequency bands × 8 binary features (energy above rolling median).

---

## Test Suite (minimal, publishable set)

### T1. Isolated vs Reactive Time Scaling (invariance vs verification)

**Hypotheses**

* **H1:** For an **isolated** process (no feedback), triad trajectories are invariant under time scaling (up to axis rescale).
* **H2:** For a **reactive** process (feedback), naive time scaling breaks triad alignment unless you re‑fit locals.

**Processes**

* **Isolated:** Poisson events (or AR(1) with no feedback to the rate).
* **Reactive:** Damped oscillator with feedback (van der Pol or driven logistic), add small Gaussian noise.

**Method**

1. Generate time series $x(t)$ length 200k, seed fixed.
2. Map to bits via D1 or D2.
3. Compute triads with $\beta$ set by half‑life $T_{1/2}$ (pre‑register e.g., 64 steps).
4. Create time‑scaled replica $x_\gamma(t) = x(t/\gamma)$, $\gamma \in \{1.2,1.5,2.0\}$.
5. Align triad curves by uniform time rescale (no cheating DTW unless declared).
6. Compute RMSE for H, C, S and a joint RMSE on the stacked vector.

**Pass/Fail**

* **Pass H1:** median joint‑RMSE$_\text{isolated}$ ≤ 0.05 across γ.
* **Pass H2:** median joint‑RMSE$_\text{reactive}$ ≥ 2× median joint‑RMSE$_\text{isolated}$ for same γ.

**Why it matters**
This cleanly tests your “continuous verification resists naive dilation” claim.

---

### T2. Pairwise Maximum‑Entropy Sufficiency (binary interactions)

**Hypothesis**

* **H3:** Pairwise max‑entropy (Ising/pseudolikelihood) captures most structure; third‑order interactions give marginal gains.

**Data**

* Use bit sequences from T1 (both isolated and reactive).
* Add one more process with known pairwise coupling: simulate a 64‑spin Ising chain (nearest‑neighbor J).

**Method**

1. Fit pairwise model via per‑spin logistic regression:
   $\Pr(s_i=1\mid s_{\neg i}) = \sigma(b_i + \sum_{j\ne i} w_{ij} s_j)$.
2. Fit an extended model with selected triple interactions (L1‑regularized).
3. Evaluate held‑out negative log‑likelihood (NLL). Compare AIC/BIC.

**Pass/Fail**

* **Pass H3:** third‑order improvement in held‑out NLL < 10% on all datasets; AIC/BIC deltas not compelling after penalty.

**Why it matters**
If pairwise suffices broadly, your “two‑body foundations” is grounded.

---

### T3. Convolutional Invariance on Band‑Limited Waves

**Hypothesis**

* **H4:** Controlled convolution/decimation preserves triad trajectories for band‑limited signals.

**Signal**

* Synthetic chirp $x(t)=\sin(2\pi f(t)t)+\epsilon$ with slowly varying $f(t)$ and SNR 20 dB. No black magic.

**Method**

1. Map to bits via D1 or D3.
2. Compute triads.
3. Apply a documented FIR low‑pass + decimate by 2; recompute triads.
4. Align by time rescale; compute RMSE.

**Pass/Fail**

* **Pass H4:** median joint‑RMSE ≤ 0.05 across 20 random seeds.
* Include a stress curve of RMSE vs decimation factor (2, 3, 4) to show the graceful fail.

**Why it matters**
This is your “convolution equals time” claim, tested under your control.

---

### T4. Retrodictive Reconstruction With Continuity Constraint

**Hypothesis**

* **H6:** Given a flip budget $k$ and a short window $W$, you can reconstruct $s_{t-1}$ from future and past triads with high accuracy.

**Setup**

* Use the reactive oscillator from T1 (harder case).
* Bit mapping D1 or D2.
* Continuity: $\mathrm{popcount}(s_t \oplus s_{t-1}) \le k$ with $k\in\{1,2,3\}$.

**Method**

1. Hold out every 5th step’s state $s_{t-1}$.
2. Use a small backward smoother that searches feasible predecessors within the flip budget.

   * Practical DP: split 64 bits into 8 blocks of 8; enumerate masks up to k within blocks; combine by additive cost (fast, exact for additive surrogates).
   * Score candidate $s_{t-1}$ by how well it matches the observed triads (C via overlap, S via flip continuity, H via EMA).
3. Choose argmin cost; measure bitwise accuracy and triad match.

**Pass/Fail**

* **Pass H6:** median bitwise accuracy ≥ 95% for $k \le 2$, $W \in [3,7]$; triad joint‑RMSE ≤ 0.05 for reconstructed vs true.

**Why it matters**
This nails your “moving forward unlocks the past” claim with numbers.

---

### T5. Smoothing Beats Filtering (uncertainty reduction)

**Hypothesis**

* **H6′:** Smoothing entropy of the latent state is lower than filtering entropy.

**Method**

1. Simulate a hidden bit‑process with known local dynamics and flip budget $k$.
2. Implement two estimators for $s_{t-1}$: filter (uses $\le t$) and smoother (uses $\le t+W$).
3. Approximate posterior entropies by enumerating feasible predecessors (or Monte Carlo when k larger).
4. Compare $H(s_{t-1}\mid \mathcal{F}_t)$ vs $H(s_{t-1}\mid \mathcal{F}_{t+W})$.

**Pass/Fail**

* **Pass H6′:** median entropy reduction ≥ 20% across windows $W\in\{3,5,7\}$.

**Why it matters**
This is the formal statement that retrodiction actually reduces uncertainty, not just vibes.

---

## Reporting & Stop Rules

**Per‑test artifacts**

* `results/Ti_metrics.csv` with per‑step H,C,S (and reconstructed H,C,S for T4/T5).
* `results/Ti_summary.json` with seeds, params, RMSEs, accuracies, CIs.
* Plots: triad overlays (before/after), reconstruction confusion matrices, sensitivity curves.

**Statistics**

* Bootstrap 10k for median and CI on RMSE/accuracy.
* Paired Wilcoxon for pre/post comparisons (H1/H2/H4/H6/H6′).
* Pre‑registered thresholds as above.

**Stop rules**

* If **T1** fails H1 **and** H2, stop. Your observables aren’t doing what you think.
* If **T3** fails H4 across seeds, rethink coherence normalization or bit mapping.
* If **T4** fails at $k\le 2$, either the smoother is wrong or the observables aren’t sufficient. Fix before proceeding.
* If **T5** shows no entropy reduction, your retrodiction story doesn’t hold.

---

## Minimal code kernels you’ll reuse

**Time scaling + triad RMSE**

```python
def triad_series(bits, beta=0.1):
    ema_flip, ema_p = 0.0, np.full(64, 0.5)
    out = []
    for i in range(1, len(bits)):
        H, C, S, ema_flip, ema_p = triad(bits[i-1], bits[i], ema_flip, ema_p, beta)
        out.append([H, C, S])
    return np.asarray(out)

def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n])**2)))
```

**FIR + decimate (controlled convolution)**

```python
from scipy.signal import firwin, lfilter

def decimate2(x, cutoff=0.45):
    taps = firwin(numtaps=63, cutoff=cutoff)  # normalized cutoff
    y = lfilter(taps, [1.0], x)[::2]
    return y
```

**Blockwise predecessor search (T4) — skeleton**

```python
from itertools import combinations

def candidate_masks_8(k):
    masks = [0]
    for j in range(1, k+1):
        for combo in combinations(range(8), j):
            m = 0
            for b in combo: m |= (1 << b)
            masks.append(m)
    return masks

BLOCKS = 8
MASKS_BY_K = {k: candidate_masks_8(k) for k in range(4)}

def reconstruct_prev_state(curr64: int, k: int, score_fn):
    """Split into 8 blocks of 8 bits; enumerate flips up to k per block; dynamic program by additive score."""
    blocks = [(curr64 >> (8*b)) & 0xFF for b in range(BLOCKS)]
    dp = [(0.0, 0)]  # (score, prev_state_bits)
    for b, curr in enumerate(blocks):
        new_dp = []
        for (score, prev_bits) in dp:
            for m in MASKS_BY_K[k]:
                prev_block = curr ^ m
                cost = score_fn(block_index=b, prev_block=prev_block, curr_block=curr)
                new_dp.append((score + cost, prev_bits | (prev_block << (8*b))))
        # keep top K beams for speed
        new_dp.sort(key=lambda t: t[0])
        dp = new_dp[:256]
    best_score, best_prev = min(dp, key=lambda t: t[0])
    return best_prev, best_score
```

You’ll plug a **triad‑consistency cost** into `score_fn` using overlap and flip penalties derived from $C$ and the continuity prior.

---

## Why this set is worth publishing

* It tests **invariance**, **pairwise sufficiency**, **convolution stability**, and **retrodiction** — the only four claims that matter.
* It’s **fully reproducible** with synthetic processes (no data gatekeeping).
* Each test has a **clear falsifier**. If it fails, you learn something real.

Run T1 → T3 → T4 → T5 in that order. If they pass, you’re not just onto “real legit math.” You’ve got a statistical‑mechanics‑of‑information story with working, verifiable evidence — in Python, no excuses.
