"""
Common utilities for SEP validation tests.
Provides triad computation, mappings, entropy estimation, and metrics.
"""

import numpy as np
from scipy import signal
from sklearn.covariance import LedoitWolf
from typing import Tuple, Dict, List, Optional, Any
import warnings

# Suppress sklearn warnings about ill-conditioned matrices
warnings.filterwarnings('ignore', category=UserWarning)

def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    
def compute_entropy(chord: np.ndarray, ema_state: Optional[np.ndarray] = None, 
                   beta: float = 0.1) -> Tuple[float, np.ndarray]:
    """
    Compute entropy using EMA of per-bit Bernoulli entropy.
    
    Args:
        chord: Binary array of shape (64,)
        ema_state: Previous EMA state, shape (64,)
        beta: EMA smoothing factor
        
    Returns:
        H: Scalar entropy value
        new_ema_state: Updated EMA state
    """
    if ema_state is None:
        ema_state = np.full(64, 0.5)
    
    # Update EMA of bit probabilities
    new_ema_state = (1 - beta) * ema_state + beta * chord
    
    # Compute per-bit entropy with numerical stability
    p = np.clip(new_ema_state, 1e-10, 1 - 1e-10)
    bit_entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    # Average entropy across bits
    H = np.mean(bit_entropy)
    
    return H, new_ema_state

def compute_coherence(chord_t: np.ndarray, chord_prev: np.ndarray,
                     density_t: float = None, density_prev: float = None) -> float:
    """
    Compute baseline-corrected coherence between two chords.
    
    Args:
        chord_t: Current chord, shape (64,)
        chord_prev: Previous chord, shape (64,)
        density_t: Optional density of current chord
        density_prev: Optional density of previous chord
        
    Returns:
        C: Coherence value
    """
    if density_t is None:
        density_t = np.mean(chord_t)
    if density_prev is None:
        density_prev = np.mean(chord_prev)
    
    # Raw overlap
    overlap = np.mean(chord_t * chord_prev)
    
    # Baseline correction
    baseline = density_t * density_prev
    max_overlap = min(density_t, density_prev)
    
    # Normalize
    if max_overlap > baseline + 1e-10:
        C = (overlap - baseline) / (max_overlap - baseline)
    else:
        C = 0.0
    
    return np.clip(C, 0, 1)

def compute_stability(chord_t: np.ndarray, chord_prev: np.ndarray,
                     ema_flips: Optional[float] = None, beta: float = 0.1) -> Tuple[float, float]:
    """
    Compute stability as 1 - EMA(flip_rate).
    
    Args:
        chord_t: Current chord, shape (64,)
        chord_prev: Previous chord, shape (64,)
        ema_flips: Previous EMA of flip rate
        beta: EMA smoothing factor
        
    Returns:
        S: Stability value
        new_ema_flips: Updated EMA of flip rate
    """
    # Compute flip rate
    flips = np.mean(chord_t != chord_prev)
    
    # Update EMA
    if ema_flips is None:
        new_ema_flips = flips
    else:
        new_ema_flips = (1 - beta) * ema_flips + beta * flips
    
    # Stability is 1 - flip_rate
    S = 1.0 - new_ema_flips
    
    return S, new_ema_flips

def compute_triad(chords: np.ndarray, beta: float = 0.1) -> Dict[str, np.ndarray]:
    """
    Compute H, C, S triad for a sequence of chords.
    
    Args:
        chords: Binary array of shape (T, 64)
        beta: EMA smoothing factor
        
    Returns:
        Dictionary with keys 'H', 'C', 'S' containing arrays of shape (T,)
    """
    T = len(chords)
    H = np.zeros(T)
    C = np.zeros(T)
    S = np.zeros(T)
    
    # Initialize EMA states
    ema_probs = np.full(64, 0.5)
    ema_flips = None
    
    for t in range(T):
        # Compute entropy
        H[t], ema_probs = compute_entropy(chords[t], ema_probs, beta)
        
        if t > 0:
            # Compute coherence
            C[t] = compute_coherence(chords[t], chords[t-1])
            
            # Compute stability
            S[t], ema_flips = compute_stability(chords[t], chords[t-1], ema_flips, beta)
        else:
            C[t] = 0.5  # Default for first timestep
            S[t] = 1.0  # Default for first timestep
    
    return {'H': H, 'C': C, 'S': S}

def mapping_D1_derivative_sign(prices: np.ndarray, lookback: int = 8) -> np.ndarray:
    """
    D1: Derivative-sign mapping (interaction-sensitive).
    Maps price derivatives and signs to 64-bit chords.
    
    Args:
        prices: Array of shape (T,) containing price values
        lookback: Number of lookback periods for derivatives
        
    Returns:
        chords: Binary array of shape (T, 64)
    """
    T = len(prices)
    chords = np.zeros((T, 64), dtype=np.float32)
    
    for t in range(T):
        # Get price window
        start_idx = max(0, t - lookback + 1)
        window = prices[start_idx:t+1]
        
        diffs = []  # Initialize diffs
        if len(window) >= 2:
            # Compute derivatives
            diffs = np.diff(window)
            
            # Map to bits (32 bits for magnitude, 32 for signs)
            for i, diff in enumerate(diffs[-32:]):
                if i < 32:
                    # Magnitude bits (thresholded)
                    chords[t, i] = float(abs(diff) > np.median(np.abs(diffs)))
                    # Sign bits
                    chords[t, 32 + i] = float(diff > 0)
        
        # Fill remaining bits with price level indicators
        price_norm = (prices[t] - np.min(prices[:t+1])) / (np.max(prices[:t+1]) - np.min(prices[:t+1]) + 1e-10)
        for i in range(len(diffs), 32):
            chords[t, i] = float(price_norm > (i / 32.0))
            
    return chords

def mapping_D2_dilation_robust(prices: np.ndarray, n_quantiles: int = 16) -> np.ndarray:
    """
    D2: Dilation-robust mapping (scale-invariant).
    Maps prices to quantile-based representation.
    
    Args:
        prices: Array of shape (T,) containing price values
        n_quantiles: Number of quantile levels
        
    Returns:
        chords: Binary array of shape (T, 64)
    """
    T = len(prices)
    chords = np.zeros((T, 64), dtype=np.float32)
    
    # Use expanding window for quantiles
    for t in range(T):
        # Get historical window
        window = prices[:t+1]
        
        if len(window) >= 2:
            # Compute quantiles
            quantiles = np.percentile(window, np.linspace(0, 100, n_quantiles + 1))
            
            # Current price position
            price = prices[t]
            
            # Binary encoding of quantile position
            for i in range(min(64, n_quantiles)):
                threshold = quantiles[min(i + 1, n_quantiles)]
                chords[t, i] = float(price >= threshold)
            
            # Additional bits encode relative position
            if t > 0:
                rel_change = (prices[t] - prices[t-1]) / (prices[t-1] + 1e-10)
                for i in range(n_quantiles, min(64, n_quantiles * 2)):
                    bit_idx = i - n_quantiles
                    threshold = -0.1 + 0.2 * (bit_idx / n_quantiles)
                    chords[t, i] = float(rel_change > threshold)
                    
            # Rolling statistics bits
            if t >= 7:
                recent = prices[t-7:t+1]
                mean = np.mean(recent)
                std = np.std(recent) + 1e-10
                z_score = (price - mean) / std
                
                for i in range(n_quantiles * 2, min(64, n_quantiles * 3)):
                    bit_idx = i - n_quantiles * 2
                    threshold = -2 + 4 * (bit_idx / n_quantiles)
                    chords[t, i] = float(z_score > threshold)
    
    return chords

def gaussian_entropy_bits(X: np.ndarray, use_ledoit_wolf: bool = True) -> float:
    """
    Estimate differential entropy assuming Gaussian distribution.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        use_ledoit_wolf: Whether to use Ledoit-Wolf covariance shrinkage
        
    Returns:
        Entropy in bits
    """
    n, d = X.shape
    
    if n <= d + 1:
        # Not enough samples for reliable estimation
        return np.log2(n) + d * np.log2(2 * np.pi * np.e)
    
    # Standardize features
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
    
    # Estimate covariance
    if use_ledoit_wolf and n > 10:
        try:
            cov, _ = LedoitWolf().fit(X_std).covariance_, None
        except:
            cov = np.cov(X_std.T)
    else:
        cov = np.cov(X_std.T)
    
    # Add regularization for numerical stability
    cov = cov + 1e-6 * np.eye(d)
    
    # Compute entropy: H = 0.5 * log(det(2πe * Σ))
    # In bits: H = 0.5 * log2(det(2πe * Σ))
    sign, logdet = np.linalg.slogdet(cov)
    
    if sign <= 0:
        # Non-positive definite, return high entropy
        return d * np.log2(2 * np.pi * np.e)
    
    entropy_nats = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
    entropy_bits = entropy_nats / np.log(2)
    
    return max(0, entropy_bits)  # Ensure non-negative

def compute_rmse(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    """Compute RMSE between two arrays."""
    return np.sqrt(np.mean((true_values - pred_values) ** 2))

def compute_joint_rmse(triads1: Dict[str, np.ndarray], 
                      triads2: Dict[str, np.ndarray]) -> float:
    """Compute joint RMSE as mean of component RMSEs."""
    rmse_h = compute_rmse(triads1['H'], triads2['H'])
    rmse_c = compute_rmse(triads1['C'], triads2['C'])
    rmse_s = compute_rmse(triads1['S'], triads2['S'])
    
    return np.mean([rmse_h, rmse_c, rmse_s])

def generate_poisson_process(rate: float, duration: float, seed: Optional[int] = None) -> np.ndarray:
    """Generate Poisson process (isolated dynamics)."""
    if seed is not None:
        np.random.seed(seed)
    
    num_events = np.random.poisson(rate * duration)
    event_times = np.sort(np.random.uniform(0, duration, num_events))
    
    # Convert to price-like signal
    dt = 0.01
    t = np.arange(0, duration, dt)
    prices = np.zeros_like(t)
    
    for event_time in event_times:
        idx = int(event_time / dt)
        if idx < len(prices):
            prices[idx:] += np.random.randn() * 0.1
    
    # Add small noise
    prices += np.random.randn(len(prices)) * 0.01
    
    return prices

def generate_van_der_pol(mu: float = 2.0, duration: float = 10.0, 
                        dt: float = 0.01, seed: Optional[int] = None) -> np.ndarray:
    """Generate Van der Pol oscillator (reactive dynamics)."""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(0, duration, dt)
    n = len(t)
    
    # Initialize
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = 2.0
    y[0] = 0.0
    
    # Integrate
    for i in range(1, n):
        dx = y[i-1] * dt
        dy = (mu * (1 - x[i-1]**2) * y[i-1] - x[i-1]) * dt
        
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
    
    # Add noise and convert to price-like signal
    prices = x + np.random.randn(n) * 0.1
    prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices)) * 2 + 1
    
    return prices

def generate_latent_coupled_processes(n_processes: int = 4, 
                                     rho: float = 0.5,
                                     length: int = 1000,
                                     seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Generate processes coupled through a latent common driver.
    
    Args:
        n_processes: Number of processes to generate
        rho: Target correlation with latent driver
        length: Length of each process
        seed: Random seed
        
    Returns:
        List of price arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate latent AR(1) driver
    phi = 0.9  # AR coefficient
    latent = np.zeros(length)
    latent[0] = np.random.randn()
    
    for t in range(1, length):
        latent[t] = phi * latent[t-1] + np.sqrt(1 - phi**2) * np.random.randn()
    
    # Generate coupled processes
    processes = []
    for _ in range(n_processes):
        # Mix latent driver with idiosyncratic noise
        idiosyncratic = np.random.randn(length)
        
        # Ensure target correlation
        process = rho * latent + np.sqrt(1 - rho**2) * idiosyncratic
        
        # Convert to price-like signal
        process = np.cumsum(process * 0.01) + 100
        processes.append(process)
    
    return processes

def time_scale_signal(signal: np.ndarray, gamma: float) -> np.ndarray:
    """
    Scale signal in time by factor gamma.
    gamma > 1: compress (speed up)
    gamma < 1: dilate (slow down)
    """
    n_original = len(signal)
    n_scaled = int(n_original / gamma)
    
    # Use interpolation for scaling
    x_original = np.linspace(0, 1, n_original)
    x_scaled = np.linspace(0, 1, n_scaled)
    
    scaled_signal = np.interp(x_scaled, x_original, signal)
    
    return scaled_signal

def apply_antialiasing_filter(signal: np.ndarray, cutoff: float = 0.4) -> np.ndarray:
    """Apply antialiasing filter before decimation."""
    # Design Butterworth filter
    b, a = signal.butter(4, cutoff, btype='low')
    filtered = signal.filtfilt(b, a, signal)
    return filtered

def safe_json_convert(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(v) for v in obj]
    else:
        return obj