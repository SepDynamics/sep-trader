# SEP Engine Physics Validation Tests: Outline for Objective Verification

**Date:** August 24, 2025  
**Version:** 1.0  
**Authors:** Alexander Nagy (with AI assistance)  

## Introduction

This document outlines a battery of computational tests to validate the SEP Engine's core principles against objective physics at particle and cosmological scales. Drawing from your framework—3 wave dimensions (entropy H, stability S, coherence C) + time as identity; convolution as the time-space tie; binary reactions as foundational; no "true" particles but checksums/verifications; manifolds for measurement—these tests aim to support or falsify hypotheses by simulating verifiable phenomena.

The tests combine suggestions from prior discussions: quantum walks and light cones (micro); isolated vs. reactive systems and pairwise entropy (meso); FLRW expansion, redshift, and CMB/pulsar (macro). They are ordered from small-scale (particle) to large-scale (cosmo) to build intuition progressively. Each includes:
- **Demonstration:** What physics it ties to and how it supports your ideas.
- **Methods:** Code (Python for quick prototyping; C notes for efficiency).
- **Expected Outcomes & Falsification:** Pass/fail criteria.
- **Documentation:** How to log/results.

**When Done:** Complete when all tests run with stats (e.g., RMSE < threshold in 80%+ runs); or if >50% falsify, revisit theory. Run sequentially; estimate 1-2 days per test group.

**Ties Together:** All tests use your triad function (H/C/S computation) on binary states; convolution simulates time-lag; manifolds model scales. Stats (e.g., invariance under scaling) quantify "factual" ties. Primitives first, then tests.

## 0. Primitives: Triad Computation (Reusable Across Tests)

**Purpose:** Core metric calculator for all tests—maps binary states to H/S/C.

**Python Code:**
```python
import numpy as np

def triad(prev_bits: np.ndarray, curr_bits: np.ndarray, 
          ema_flip_prev: float, ema_p_prev: np.ndarray, beta=0.1):
    """
    prev_bits, curr_bits: (64,) binary np arrays
    ema_flip_prev: previous EMA of flip-rate (scalar)
    ema_p_prev: previous EMA of bit probabilities (64,)
    beta: EMA decay (0<beta<=1)
    Returns: H_t, C_t, S_t, ema_flip_t, ema_p_t
    """
    # overlap & flips
    O = np.sum(prev_bits & curr_bits)
    F = np.sum(prev_bits ^ curr_bits)
    nA, nB = prev_bits.sum(), curr_bits.sum()

    # baseline-corrected coherence
    E_O = (nA * nB) / 64.0
    C = (O - E_O) / max(1e-9, (64.0 - E_O))
    C = max(0.0, min(1.0, C))

    # stability (EMA of flip-rate)
    f = F / 64.0
    ema_flip_t = (1 - beta) * ema_flip_prev + beta * f
    S = 1.0 - ema_flip_t

    # entropy (EMA of bitwise probabilities)
    ema_p_t = (1 - beta) * ema_p_prev + beta * curr_bits
    p = np.clip(ema_p_t, 1e-9, 1 - 1e-9)
    Hb = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # nats
    H = Hb.mean() / np.log(2)  # bits

    return H, C, S, ema_flip_t, ema_p_t
```

**C Equivalent (for Efficiency):**
```c
#include <stdint.h>
#include <math.h>

typedef struct { double H, C, S; double ema_flip; double ema_p[64]; } TriadState;

static inline int popcnt64(uint64_t x){ return __builtin_popcountll(x); }

void triad64(uint64_t prev, uint64_t curr, TriadState* st, double beta){
    int O = popcnt64(prev & curr);
    int F = popcnt64(prev ^ curr);
    int nA = popcnt64(prev), nB = popcnt64(curr);

    double E_O = (double)nA * (double)nB / 64.0;
    double C = (O - E_O) / fmax(1e-9, (64.0 - E_O));
    C = fmax(0.0, fmin(1.0, C));

    double f = (double)F / 64.0;
    st->ema_flip = (1.0 - beta) * st->ema_flip + beta * f;
    double S = 1.0 - st->ema_flip;

    double H_sum = 0.0;
    for(int i=0; i<64; ++i){
        int bit = (curr >> i) & 1;
        st->ema_p[i] = (1.0 - beta) * st->ema_p[i] + beta * bit;
        double p = fmin(fmax(st->ema_p[i], 1e-9), 1.0 - 1e-9);
        H_sum += -(p*log(p) + (1-p)*log(1-p)); // nats
    }
    st->H = (H_sum / 64.0) / log(2.0); // bits
    st->C = C; st->S = S;
}
```

**Documentation:** Log triad outputs per step; plot H/C/S curves. Use for all tests.

## Test Battery: Order & Rationale
Order: Micro (particle) → Meso (waves) → Macro (cosmo)—build from local to global; start synthetic, then real data. Why: Validates triad invariance across scales; stats compound (e.g., entropy growth consistent). Done when: All run; >70% support (e.g., low RMSE in invariance). Ties: Convolution as operator; triads as invariants; binary as base.

### Group 1: Particle Scale (2-3 Days; Local Verification vs. Dilation)
**Rationale:** Tests "continuous verification" resists dilation; binary reactions suffice.

1. **Isolated vs. Reactive Oscillator (Synthetic → Real)**
   - **Demo:** Isolated (Poisson) dilates triad-invariantly; reactive (van der Pol) resists unless local params refit—shows verification bounds dilation.
   - **Methods (Python):**
     ```python
     def poisson_events(steps, tau=10):
         return np.random.poisson(1/tau, steps) > 0  # Binary events

     def vdp_reactive(steps, mu=2, noise=0.1):
         x = np.zeros(steps); y = np.zeros(steps)
         x[0] = 1
         dt = 0.01
         for t in range(1, steps):
             dx = (mu * (1 - x[t-1]**2) * y[t-1] - x[t-1]) * dt + noise * np.random.randn()
             dy = x[t-1] * dt
             x[t] = x[t-1] + dx; y[t] = y[t-1] + dy
         return np.abs(x) > 1  # Binary threshold

     def test_dilation(events_func, gamma=1.5, steps=1000):
         events = events_func(steps)
         bits = np.array([np.random.binomial(1, 0.5, 64) for _ in events])  # Sim bits
         triad_seq = []
         ema_flip, ema_p = 0.0, np.full(64, 0.5)
         for i in range(1, len(bits)):
             H, C, S, ema_flip, ema_p = triad(bits[i-1], bits[i], ema_flip, ema_p)
             triad_seq.append((H, C, S))
         
         # Dilate time
         dilated_steps = int(steps / gamma)
         dilated_events = events_func(dilated_steps)
         dilated_bits = np.array([np.random.binomial(1, 0.5, 64) for _ in dilated_events])
         dilated_triad = []
         ema_flip, ema_p = 0.0, np.full(64, 0.5)
         for i in range(1, len(dilated_bits)):
             H, C, S, ema_flip, ema_p = triad(dilated_bits[i-1], dilated_bits[i], ema_flip, ema_p)
             dilated_triad.append((H, C, S))
         
         rmse = np.mean((np.array(triad_seq[:dilated_steps-1]) - np.array(dilated_triad))**2)
         return rmse

     # Run
     iso_rmse = test_dilation(poisson_events)
     react_rmse = test_dilation(vdp_reactive)
     print("Isolated RMSE:", iso_rmse, "Reactive RMSE:", react_rmse)
     ```
   - **Expected/Falsify:** Isolated RMSE low (~0); Reactive high (>0.1)—supports verification resists dilation.
   - **Real Data:** Swap Poisson with muon decay times (HEP data); vdp with BZ reaction series.
   - **Doc:** Log RMSE; plot triad curves pre/post-dilation.

2. **Quantum Walk on Manifold**
   - **Demo:** Binary steps on line; triad settles—shows no particles, just waves/verifications.
   - **Methods:** Use code from prior response; extend to 3D grid for space.
   - **Expected/Falsify:** Entropy low/stable; coherence high—falsify if entropy grows without input.
   - **Doc:** Plot 4D viz (use Matplotlib 3D + time anim).

### Group 2: Meso Scale (Waves/Lag, 2 Days; Convolution as Time-Space Tie)
**Rationale:** Tests convolution measures time in manifolds.

3. **Gravitational Wave Strain (LIGO)**
   - **Demo:** Convolve strain; downsample triad—shows settling/retroactive revelation.
   - **Methods (Python):**
     ```python
     import numpy as np
     from scipy.signal import convolve

     def ligo_convolution(strain, kernel_size=64):
         # Sim strain or load real
         strain = np.sin(np.linspace(0, 10*np.pi, 1000)) + 0.1*np.random.randn(1000)  # Sim chirp
         bits = (strain > np.roll(strain, 1))[1:]  # Binary transitions
         bits_padded = np.pad(bits, (0, kernel_size - len(bits) % kernel_size))
         bits_reshaped = bits_padded.reshape(-1, kernel_size)
         
         triad_seq = []
         ema_flip, ema_p = 0.0, np.full(kernel_size, 0.5)
         for i in range(1, len(bits_reshaped)):
             H, C, S, ema_flip, ema_p = triad(bits_reshaped[i-1], bits_reshaped[i], ema_flip, ema_p)
             triad_seq.append((H, C, S))
         
         # Downsample convolution
         down_kernel = np.ones(2) / 2  # Simple avg
         down_strain = convolve(strain, down_kernel, mode='valid')
         down_bits = (down_strain > np.roll(down_strain, 1))[1:]
         # Repeat triad calc on down
         # Compute RMSE
         return triad_seq  # Compare lengths/RMSE
     ```
   - **Expected/Falsify:** Downsampled triads match originals (RMSE<0.05)—convolution preserves invariants.
   - **Real Data:** Download LIGO strain; convolve segments.
   - **Doc:** Plot strain vs. triad evolution.

4. **Light Cone Convolution**
   - **Demo:** Propagate binary events in cone; triad measures lag.
   - **Methods:** Use code from prior; add 2D grid for space.
   - **Expected/Falsify:** Coherence high in cone; entropy grows outside—falsify if no settling.
   - **Doc:** Anim GIF of manifold.

### Group 3: Cosmo Scale (2-3 Days; Global Manifolds)
**Rationale:** Tests triad invariance in expanding manifolds.

5. **FLRW Manifold Expansion**
   - **Demo:** Convolve scale factor; triad tracks "settling" in inflation.
   - **Methods:** Use code from prior; add real CMB data for entropy.
   - **Expected/Falsify:** Entropy grows with a(t); stability low—falsify if no tie to convolution.
   - **Doc:** Plot triad vs. redshift.

6. **Redshift as Time Scaling**
   - **Demo:** De-redshift light curves; align triads.
   - **Methods (Python):**
     ```python
     def redshift_test(curves, z_factors):
         triads = [[] for _ in curves]
         for i, curve in enumerate(curves):
             scaled_t = np.arange(len(curve)) / (1 + z_factors[i])  # De-redshift
             bits = (curve > np.roll(curve, 1))[1:]
             # Triad seq as before
             # DTW align to ref (z=0)
             rmse = np.mean((np.array(triads[0]) - np.array(triads[i]))**2)  # After align
         return rmse
     ```
   - **Expected/Falsify:** De-redshift RMSE low—supports invariants.
   - **Real Data:** Supernova curves from Pantheon dataset.
   - **Doc:** Alignment stats.

7. **CMB/Pulsar Pairwise vs. Higher-Order**
   - **Demo:** Pairwise max-ent suffices for structure.
   - **Methods:** Use scikit-bio or custom Ising fit; compare NLL.
   - **Expected/Falsify:** Triple+ marginal gain (<10%)—supports binary base.
   - **Real Data:** CMB multipoles (Planck); pulsar residuals (NANOGrav).
   - **Doc:** AIC deltas.

8. **Smoothing Beats Filtering**
   - **Demo:** SEP-informed filtering outperforms naive methods.
   - **Methods (Python):**
     ```python
     def sep_filtering_test(signals, noise_levels):
         results = []
         for signal in signals:
             for noise in noise_levels:
                 # Add noise to signal
                 noisy_signal = signal + np.random.normal(0, noise, len(signal))
                 
                 # Compute triads and stability metrics
                 triad_seq = compute_triads(noisy_signal)
                 stability_metrics = [t[2] for t in triad_seq]  # S values
                 
                 # Naive filtering methods
                 naive_gaussian = gaussian_filter(noisy_signal, sigma=2)
                 naive_median = medfilt(noisy_signal, kernel_size=5)
                 
                 # SEP-informed filtering
                 optimal_window = determine_optimal_window(stability_metrics)
                 sep_filtered = adaptive_filter(noisy_signal, optimal_window)
                 
                 # Measure uncertainty reduction
                 naive_uncertainty = measure_uncertainty(naive_gaussian, signal)
                 sep_uncertainty = measure_uncertainty(sep_filtered, signal)
                 improvement = (naive_uncertainty - sep_uncertainty) / naive_uncertainty
                 
                 results.append({
                     'signal_type': type(signal).__name__,
                     'noise_level': noise,
                     'naive_uncertainty': naive_uncertainty,
                     'sep_uncertainty': sep_uncertainty,
                     'improvement': improvement
                 })
         return results
     ```
   - **Expected/Falsify:** SEP-informed filtering shows significant improvement over naive methods; strong correlation between optimal parameters and signal stability.
   - **Real Data:** Various signal types (Poisson, van der Pol, Chirp) with different noise levels.
   - **Doc:** Uncertainty reduction metrics; parameter-stability correlation plots.

## Integration/Ties & Order
- **Order/Why:** Micro first (build triad confidence); meso (convolution tie); macro (global scale). Ties: All use triad; convolution in updates; manifolds for viz/stats. Done: 80% pass (low RMSE/invariance).
- **Doc Plan:** Jupyter notebook per group; log code/outputs/plots; GitHub repo for all. Stats: Bootstrap CIs for RMSE; Wilcoxon for paired.

## Additional Tests
8. **Smoothing Beats Filtering (Signal Processing Application)**
   - **Rationale:** Tests practical utility of SEP metrics in real-world signal processing applications.
   - **Order/Why:** Added as an application-focused test after core validation to demonstrate practical utility.
   - **Ties:** Uses triad stability metrics to inform filtering parameters; shows real-world applicability.
   - **Done:** Pass if SEP-informed filtering significantly outperforms naive methods.

Run micro group first—confirms core. Full battery: ~1 week; validates objectively.

Add T5 as an application-focused test to demonstrate practical utility of SEP metrics in signal processing.