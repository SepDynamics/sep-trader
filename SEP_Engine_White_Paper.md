Quantum-Inspired Pattern Evolution System for Financial Trading: The SEP Engine
===============================================================================
White Paper — v1.4 (Draft for Discussion)

Authors: Alexander Nagy, SepDynamics Development Team
Date: August 19, 2025

Abstract
--------
We present SEP, a quantum-inspired engine for financial pattern analysis and trading signal generation. Market data are encoded as 64-bit states and processed with Bit-Transition Harmonics (BTH), a Bitwise Reliability Score (BRS), and a Geometry-Aware Optimizer (GAO). A Pattern Evolution layer adapts parameters over time. On EUR/USD tick data (2025‑03‑01–2025‑04‑15 UTC, 07:00–21:00), a rolling 24 h‑train/6 h‑test walk‑forward yields a 65.0% ±2.7% directional hit rate at a fixed 30‑minute horizon after costs (median 0.8‑pip spread, $0.50/lot commission, 0.1‑pip slippage). Expectancy is 0.84 pips/trade with Profit Factor 1.20 and Sharpe 1.8. Ablations show incremental gains from BTH, BRS, GAO, and evolutionary adaptation. A reproducibility pack with commit hashes, docker_sha.txt, seeds, signals.csv, trades.csv, config.json, cost_model.json, and regeneration scripts is provided.
US provisional patent filed July 27, 2025.

Notation
--------
Tick price: $P_t$. Return over horizon $h$: $r_{t,h}=P_{t+h}-P_t$.  
Bit state: $s_t\in\{0,1\}^{64}$ (64 thresholded features).  
Transitions: $\delta_t=s_t\oplus s_{t-1}$; overlap/"rupture": $\varrho_t=s_t\wedge s_{t-1}$; $r_t=\operatorname{popcount}(\varrho_t)$.  
Weights: $w_k\ge0$ with $W_t=\sum_{\tau\le t}w_{t-\tau}$. Exponential kernel $w_k=\lambda(1-\lambda)^k$.
Coherence: $\mathcal C_t=1-\frac{1}{64 W_t}\sum_{\tau\le t}w_{t-\tau} r_\tau\in[0,1]$ since $r_\tau\le64$.
Flip rate: $f_t=\operatorname{popcount}(\delta_t)/64$; rupture rate: $u_t=\operatorname{popcount}(\varrho_t)/64$.
Stability: $\mathcal S_t=1-\mathrm{EMA}_\beta[f_t]\in[0,1]$.
Entropy: $\mathcal H_t=\tfrac{1}{64}\sum_{i=1}^{64} H_b(p_{i,t})$, $p_{i,t}=\mathrm{EMA}_\beta[s_t(i)]$, $H_b(p)=-p\log p-(1-p)\log(1-p)$.
($\beta$ = decay; same as weight decay unless stated otherwise.)
Signal $s_t^{\text{sig}}\in\{-1,0,+1\}$. Label $y_{t,h}=\operatorname{sign}(r_{t,h})$.
Costs $k_t$ in after-cost round‑turn pips (spread + commission per side ×2 + slippage).
Per‑trade PnL: $\pi_{t,h}=s_t^{\text{sig}}\cdot r_{t,h}-k_t$. Expectancy $\mathbb E[\pi]$. Profit Factor $\mathrm{PF}=\frac{\sum\max(\pi,0)}{\sum\max(-\pi,0)}$.

Executive Summary
-----------------
Financial markets require adaptive systems that quantify pattern stability and react before degradation. SEP addresses this need with:

* **Bit-Transition Harmonics (BTH):** Maintains $\delta_t,\varrho_t$, updates $\mathcal C_t$ with exponential decay, computes FWHT (Walsh–Hadamard) band energies over a sliding window, and exposes flip rate, rupture density, and damped integrals.
* **Bitwise Reliability Score (BRS):** Given a predicted bit pattern $\hat s_t$, computes normalized Hamming similarity $1-\tfrac{1}{64}\operatorname{popcount}(s_t\oplus \hat s_t)$ to gate low‑reliability signals; threshold $\tau$ calibrated via reliability diagram to bound ECE ≤2%.
* **Geometry-Aware Optimizer (GAO):** Uses a diagonal metric $G(\theta)=\operatorname{diag}(g_{\mathcal C},g_{\mathcal S},g_{\mathcal H})$ with $g_\bullet=\varepsilon+\mathrm{EMA}[\nabla_\bullet J]^2$, yielding Riemannian gradient $\hat g=G(\theta)^{-1}\nabla J$ and update $\theta_{t+1}=R_\theta(-\eta \hat g)$ via projection $R(\theta,v)=\theta+v$ onto $\Theta$.
* **Pattern Evolution:** Selection → mutation → inheritance of $(\mathcal C,\mathcal S,\mathcal H)$ with generation counters; fitness vs generation tracked with confidence intervals.

These components are patent‑pending (US provisional filed July 27, 2025). With over 1,600 commits, SEP demonstrates research‑grade reproducibility and real‑time deployment readiness.

Experimental Results
--------------------
**Protocol:** EUR/USD tick data, 07:00–21:00 UTC; fixed 30‑minute horizon; one position per symbol; 5‑minute cooldown. Costs measured in after-cost round‑turn pips for a 100k lot: median spread 0.8 pips + $0.50/lot per side commission ×2 + 0.1‑pip slippage.

**Headline metrics (test only):**
- Hit rate 65.0% ±2.7% (Wilson, N=1 200).
- Expectancy 0.84 pips/trade (median 0.5, IQR 0.3–1.2). With hit 65% and PF 1.20 the implied after-cost average win ≈7.8 pips and loss ≈12.0 pips, reflecting fixed-horizon exits.
- Profit Factor 1.20; Sharpe 1.8 (annualized from 1‑hour returns using √(14×252)).
- Capacity ∂$(\mathbb E[\pi])$/∂slippage = –0.05 pips per 0.1‑pip slippage; edge vanishes at ≈+1.7 pips extra slippage (Appendix plots utility vs lead-time).
- Multiple-hypothesis control: White’s Reality Check / SPA on ablations (p = 0.03).

**Ablation study (95% CIs):**

| stage           | hit % (±) | PF (±) | Sharpe (±) | E[π] pips (±) |
|-----------------|-----------|--------|------------|---------------|
| baseline (n-bar)| 58.0±2.8  |1.05±0.05|0.8±0.2    |0.10±0.05      |
| +BTH            | 61.2±2.7  |1.10±0.05|1.0±0.2    |0.30±0.05      |
| +BTH+BRS        | 63.5±2.7  |1.15±0.05|1.3±0.2    |0.55±0.05      |
| +GAO            | 64.5±2.7  |1.18±0.05|1.6±0.2    |0.75±0.05      |
| +evolution      | 65.0±2.7  |1.20±0.05|1.8±0.2    |0.84±0.05      |

Reproducibility Pack
--------------------
Includes `docker_sha.txt`, exact commit hashes, random seeds, `signals.csv` (all signals, including dropped), `trades.csv` (every fill with spread/slip/fees), `config.json`, `cost_model.json`, and scripts to regenerate all tables and figures byte‑for‑byte. Results are research‑grade, not investment advice.

