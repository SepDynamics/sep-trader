#!/usr/bin/env python3
"""
SEP-InfoStat Core Implementation
Core triad observables (H, C, S) and bit mapping functions
"""

import numpy as np
from typing import Tuple, List
from scipy.signal import firwin, lfilter
from itertools import combinations

# Global settings
RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)

def triad(prev_bits: np.ndarray, curr_bits: np.ndarray,
          ema_flip_prev: float, ema_p_prev: np.ndarray, beta: float) -> Tuple[float, float, float, float, np.ndarray]:
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

def triad_series(bits: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """Compute triad series from bit sequence."""
    ema_flip, ema_p = 0.0, np.full(64, 0.5)
    out = []
    for i in range(1, len(bits)):
        H, C, S, ema_flip, ema_p = triad(bits[i-1], bits[i], ema_flip, ema_p, beta)
        out.append([H, C, S])
    return np.asarray(out)

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSE between two arrays."""
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n])**2)))

# Bit mapping functions
def bit_mapping_D1(signal: np.ndarray, window_size: int = 64) -> np.ndarray:
    """D1: Sign of derivative over 64 staggered micro-windows."""
    n = len(signal)
    bits_sequence = []
    
    for i in range(window_size, n - window_size):
        bits = np.zeros(64, dtype=int)
        for j in range(64):
            start_idx = i + j * (window_size // 64)
            if start_idx + 1 < n:
                derivative = signal[start_idx + 1] - signal[start_idx]
                bits[j] = 1 if derivative > 0 else 0
        bits_sequence.append(bits)
    
    return np.array(bits_sequence)

def bit_mapping_D2(signal: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """D2: 64 rolling quantile thresholds."""
    n = len(signal)
    quantiles = np.linspace(0.015, 0.985, 64)  # 1.5% to 98.5%
    bits_sequence = []
    
    for i in range(window_size, n):
        window = signal[i-window_size:i]
        thresholds = np.quantile(window, quantiles)
        bits = (signal[i] > thresholds).astype(int)
        bits_sequence.append(bits)
    
    return np.array(bits_sequence)

def bit_mapping_D3(signal: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """D3: 8 frequency bands × 8 binary features (energy above rolling median)."""
    from scipy.signal import welch
    n = len(signal)
    bits_sequence = []
    
    for i in range(window_size, n, window_size // 8):  # Overlap windows
        if i + window_size > n:
            break
        window = signal[i:i+window_size]
        
        # Compute power spectral density
        freqs, psd = welch(window, nperseg=128)
        
        # Split into 8 frequency bands
        band_size = len(psd) // 8
        bits = np.zeros(64, dtype=int)
        
        for band in range(8):
            start_f = band * band_size
            end_f = (band + 1) * band_size if band < 7 else len(psd)
            band_energy = np.sum(psd[start_f:end_f])
            
            # Create 8 binary features per band (different thresholds)
            for feature in range(8):
                # Rolling median reference
                if len(bits_sequence) > 10:
                    recent_energies = [bits_sequence[k][band * 8 + feature] for k in range(max(0, len(bits_sequence)-10), len(bits_sequence))]
                    threshold = np.median(recent_energies) if recent_energies else 0.5
                else:
                    threshold = 0.5
                
                bits[band * 8 + feature] = 1 if band_energy > threshold else 0
        
        bits_sequence.append(bits)
    
    return np.array(bits_sequence)

# Signal generation utilities
def generate_poisson_process(length: int, rate: float = 1.0, seed: int = None) -> np.ndarray:
    """Generate isolated Poisson process."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate inter-arrival times
    inter_arrivals = np.random.exponential(1/rate, length*2)
    arrival_times = np.cumsum(inter_arrivals)
    
    # Create signal
    signal = np.zeros(length)
    t = np.arange(length)
    
    for arrival in arrival_times:
        if arrival < length:
            idx = int(arrival)
            if idx < length:
                signal[idx] = 1.0
    
    # Add cumulative effect
    signal = np.cumsum(signal) + 0.1 * np.random.randn(length)
    return signal

def generate_van_der_pol(length: int, mu: float = 1.0, dt: float = 0.01, seed: int = None) -> np.ndarray:
    """Generate reactive van der Pol oscillator with feedback."""
    if seed is not None:
        np.random.seed(seed)
    
    x, y = 1.0, 0.0
    signal = np.zeros(length)
    
    for i in range(length):
        # Van der Pol dynamics with small noise
        dx = y
        dy = mu * (1 - x*x) * y - x
        
        # Add noise
        dx += 0.01 * np.random.randn()
        dy += 0.01 * np.random.randn()
        
        x += dt * dx
        y += dt * dy
        signal[i] = x
    
    return signal

def generate_chirp(length: int, f0: float = 10, f1: float = 50, snr_db: float = 20, seed: int = None) -> np.ndarray:
    """Generate synthetic chirp signal for convolution testing."""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, 1, length)
    # Linear chirp
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2)
    signal = np.sin(phase)
    
    # Add noise
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(length)
    
    return signal + noise

# FIR filtering utilities
def decimate2(x: np.ndarray, cutoff: float = 0.45) -> np.ndarray:
    """FIR + decimate by 2 (controlled convolution)."""
    taps = firwin(numtaps=63, cutoff=cutoff)  # normalized cutoff
    y = lfilter(taps, [1.0], x)[::2]
    return y

# Reconstruction utilities for T4
def candidate_masks_8(k: int) -> List[int]:
    """Generate candidate flip masks for 8-bit blocks."""
    masks = [0]
    for j in range(1, k+1):
        for combo in combinations(range(8), j):
            m = 0
            for b in combo: 
                m |= (1 << b)
            masks.append(m)
    return masks

BLOCKS = 8
MASKS_BY_K = {k: candidate_masks_8(k) for k in range(4)}

def reconstruct_prev_state(curr64: int, k: int, score_fn) -> Tuple[int, float]:
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
        # Keep top 256 beams for speed
        new_dp.sort(key=lambda t: t[0])
        dp = new_dp[:256]
    
    best_score, best_prev = min(dp, key=lambda t: t[0])
    return best_prev, best_score

def bits_to_int(bits: np.ndarray) -> int:
    """Convert 64-bit array to integer."""
    return int(''.join(map(str, bits.astype(int))), 2)

def int_to_bits(value: int) -> np.ndarray:
    """Convert integer to 64-bit array."""
    bits_str = format(value, '064b')
    return np.array([int(b) for b in bits_str])

# Additional utility functions for test support
def controlled_convolution_decimation(signal: np.ndarray, decimation_factor: int = 2) -> np.ndarray:
    """Apply controlled convolution and decimation for testing invariance."""
    from scipy.signal import firwin, lfilter
    
    # Design anti-aliasing filter
    nyquist = 0.5
    cutoff = nyquist / decimation_factor
    taps = firwin(numtaps=63, cutoff=cutoff, window='hamming')
    
    # Apply filter
    filtered = lfilter(taps, [1.0], signal)
    
    # Decimate
    decimated = filtered[::decimation_factor]
    
    return decimated

def compute_joint_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Compute joint entropy of two discrete variables."""
    # Discretize if continuous
    if x.dtype.kind == 'f' or y.dtype.kind == 'f':
        # Convert to discrete bins
        x_bins = np.digitize(x, bins=np.linspace(x.min(), x.max(), 10))
        y_bins = np.digitize(y, bins=np.linspace(y.min(), y.max(), 10))
    else:
        x_bins = x
        y_bins = y
    
    # Compute joint distribution
    joint_counts = np.histogram2d(x_bins, y_bins, bins=10)[0]
    joint_probs = joint_counts / joint_counts.sum()
    
    # Remove zeros to avoid log(0)
    joint_probs = joint_probs[joint_probs > 0]
    
    # Compute entropy
    return -np.sum(joint_probs * np.log2(joint_probs))