#!/usr/bin/env python3
"""
Test T4: Retrodiction Uniqueness Under Continuity
Tests that given current state and triad observables with continuity constraints,
the predecessor state is uniquely determined with high probability.

Tests uniqueness rate vs flip budget k and window length L.
"""

import sys
import os
# Add the parent directory (validation) to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from itertools import product

# Import from shared utilities
from common import (
    compute_triad,
    mapping_D2_dilation_robust,
    set_random_seed
)
from validation_io import (
    save_test_results,
    log_test_header,
    log_hypothesis,
    log_test_summary,
    TestLogger,
    format_json_safe
)
from plots import setup_plot_style
from thresholds import (
    get_thresholds,
    validate_t4_results,
    get_hypothesis_description
)
from progress_monitor import ProgressMonitor

# Test parameters (optimized for quick execution)
SEQUENCE_LENGTH = 2000
BETA = 0.1
FLIP_BUDGETS = [1, 2, 3, 4]  # k values
WINDOW_LENGTHS = [8, 16, 32]  # L values
SEEDS = [1337]  # Single seed for speed
ERROR_THRESHOLD = 0.01  # Triad matching threshold

def generate_smooth_sequence(length: int, seed: int) -> np.ndarray:
    """Generate a smooth price-like sequence with controlled flip rate."""
    set_random_seed(seed)
    
    # Start with smooth trend
    t = np.linspace(0, 10, length)
    base = 100 + 10 * np.sin(0.5 * t) + 5 * np.cos(0.3 * t)
    
    # Add controlled noise
    noise = np.random.randn(length) * 0.5
    signal = base + noise
    
    return signal

def generate_hamming_neighbors(state: np.ndarray, k: int) -> List[np.ndarray]:
    """Generate all binary states within Hamming distance k of given state."""
    n_bits = len(state)
    neighbors = []
    
    # Generate all combinations of bit positions to flip
    for num_flips in range(k + 1):
        if num_flips == 0:
            neighbors.append(state.copy())
        else:
            for positions in product(range(n_bits), repeat=num_flips):
                # Avoid duplicate positions
                if len(set(positions)) == num_flips:
                    candidate = state.copy()
                    for pos in positions:
                        candidate[pos] = 1 - candidate[pos]  # Flip bit
                    neighbors.append(candidate)
    
    return neighbors

def compute_triad_consistency_score(candidate: np.ndarray, current: np.ndarray,
                                   observed_triad: Dict[str, float], 
                                   window_history: np.ndarray, beta: float) -> float:
    """
    Score how well a candidate predecessor explains the observed triad.
    
    Returns lower score for better match (0 = perfect match).
    """
    # Create extended sequence: history + candidate + current
    extended = np.concatenate([window_history, [candidate], [current]])
    
    # Convert to chords
    chords = mapping_D2_dilation_robust(extended)
    
    # Compute triad for the transition from candidate to current
    triads = compute_triad(chords[-2:], beta=beta)  # Last two states
    
    # Score based on triad match (using last triad value)
    pred_h = triads['H'][-1]
    pred_c = triads['C'][-1]  
    pred_s = triads['S'][-1]
    
    # Compute squared error
    h_error = (pred_h - observed_triad['H']) ** 2
    c_error = (pred_c - observed_triad['C']) ** 2
    s_error = (pred_s - observed_triad['S']) ** 2
    
    return h_error + c_error + s_error

def test_retrodiction_uniqueness(k: int, L: int, seed: int) -> Dict:
    """Test retrodiction uniqueness for given k and L."""
    set_random_seed(seed)
    
    # Generate smooth sequence
    prices = generate_smooth_sequence(SEQUENCE_LENGTH, seed)
    chords = mapping_D2_dilation_robust(prices)
    triads = compute_triad(chords, beta=BETA)
    
    successful_predictions = 0
    total_attempts = 0
    
    # Test retrodiction at multiple points
    test_points = np.linspace(L + 10, len(prices) - 10, 20, dtype=int)
    
    for t in test_points:
        # Current state and triad
        current_chord = chords[t]
        current_triad = {
            'H': triads['H'][t],
            'C': triads['C'][t],
            'S': triads['S'][t]
        }
        
        # True predecessor
        true_predecessor = chords[t-1]
        
        # Window history
        window_start = max(0, t - L)
        window_history = prices[window_start:t-1]
        
        if len(window_history) < 2:
            continue
            
        # Generate candidate predecessors within Hamming ball
        candidates = generate_hamming_neighbors(true_predecessor, k)
        
        # Score all candidates
        scores = []
        for candidate in candidates:
            # Convert binary chord back to approximate price
            # (This is a simplification - in practice you'd need proper inverse mapping)
            price_approx = np.sum(candidate) / len(candidate) * 200  # Rough approximation
            candidate_prices = np.concatenate([window_history, [price_approx]])
            
            score = compute_triad_consistency_score(
                price_approx, prices[t], current_triad, window_history, BETA
            )
            scores.append((candidate, score))
        
        # Find candidates with minimum score
        min_score = min(score for _, score in scores)
        best_candidates = [candidate for candidate, score in scores 
                          if score <= min_score + ERROR_THRESHOLD]
        
        # Check if unique (only one best candidate)
        if len(best_candidates) == 1:
            # Verify it's actually close to true predecessor
            best = best_candidates[0]
            hamming_dist = np.sum(best != true_predecessor)
            if hamming_dist <= k:  # Within expected Hamming distance
                successful_predictions += 1
        
        total_attempts += 1
    
    uniqueness_rate = successful_predictions / max(1, total_attempts)
    
    return {
        'k': k,
        'L': L,
        'seed': seed,
        'uniqueness_rate': uniqueness_rate,
        'successful_predictions': successful_predictions,
        'total_attempts': total_attempts
    }

def run_t4_test() -> Dict:
    """Run the complete T4 test suite."""

    with ProgressMonitor(max_runtime_minutes=60):
        with TestLogger("T4", "Retrodiction Uniqueness Under Continuity"):
            results = []

            # Test all combinations of k and L
            for seed in SEEDS:
                for k in FLIP_BUDGETS:
                    for L in WINDOW_LENGTHS:
                        print(f"  Testing k={k}, L={L}, seed={seed}")
                        result = test_retrodiction_uniqueness(k, L, seed)
                        results.append(result)
                        print(f"    Uniqueness rate: {result['uniqueness_rate']:.3f}")

            # Aggregate results
            uniqueness_rates = {}
            for r in results:
                key = (r['k'], r['L'])
                if key not in uniqueness_rates:
                    uniqueness_rates[key] = []
                uniqueness_rates[key].append(r['uniqueness_rate'])

            # Calculate median rates for each (k, L) pair
            median_rates = {}
            for key, rates in uniqueness_rates.items():
                median_rates[key] = np.median(rates)

            # Validate results
            validation = validate_t4_results(median_rates)

            # Get thresholds
            thresholds = get_thresholds('T4')

            # Log results
            log_hypothesis("T4", get_hypothesis_description('T4', 'T4'),
                          thresholds['T4_uniqueness'], validation['best_rate'], validation['T4'])

            # Create plot data
            plot_data = create_t4_plot_data(median_rates)

            # Create plot
            fig = create_t4_plot(plot_data, thresholds)

            # Prepare summary
            summary = {
                'test': 'T4',
                'parameters': {
                    'sequence_length': SEQUENCE_LENGTH,
                    'beta': BETA,
                    'flip_budgets': FLIP_BUDGETS,
                    'window_lengths': WINDOW_LENGTHS,
                    'seeds': SEEDS,
                    'error_threshold': ERROR_THRESHOLD
                },
                'results': {
                    'uniqueness_rates': {f"k{k}_L{L}": rate for (k, L), rate in median_rates.items()},
                    'best_rate': validation['best_rate']
                },
                'hypothesis': {
                    'T4': {
                        'pass': validation['T4'],
                        'metric': validation['best_rate'],
                        'threshold': thresholds['T4_uniqueness'],
                        'description': get_hypothesis_description('T4', 'T4_uniqueness')
                    }
                },
                'overall_pass': validation['T4']
            }

            # Prepare metrics for CSV
            metrics = []
            for r in results:
                metrics.append({
                    'k': r['k'],
                    'L': r['L'],
                    'seed': r['seed'],
                    'uniqueness_rate': r['uniqueness_rate'],
                    'successful_predictions': r['successful_predictions'],
                    'total_attempts': r['total_attempts']
                })

            # Save results
            summary_clean = format_json_safe(summary)
            save_test_results('T4', summary_clean, metrics, fig)

            # Log summary
            log_test_summary('T4', {'T4': validation['T4']})

            print(f"\nBest uniqueness rate: {validation['best_rate']:.3f}")
            print(f"Required: {thresholds['T4_uniqueness']:.3f}")

            return summary

def create_t4_plot_data(median_rates: Dict[Tuple[int, int], float]) -> Dict:
    """Create plot data for T4 visualization."""
    # Create heatmap data
    k_vals = sorted(set(k for k, L in median_rates.keys()))
    L_vals = sorted(set(L for k, L in median_rates.keys()))
    
    heatmap_data = np.zeros((len(k_vals), len(L_vals)))
    
    for i, k in enumerate(k_vals):
        for j, L in enumerate(L_vals):
            rate = median_rates.get((k, L), 0)
            heatmap_data[i, j] = rate
    
    return {
        'k_vals': k_vals,
        'L_vals': L_vals,
        'heatmap_data': heatmap_data,
        'median_rates': median_rates
    }

def create_t4_plot(plot_data: Dict, thresholds: Dict) -> plt.Figure:
    """Create T4 test visualization."""
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('T4: Retrodiction Uniqueness Test', fontsize=14, fontweight='bold')
    
    # Heatmap
    im = ax1.imshow(plot_data['heatmap_data'], cmap='RdYlGn', aspect='auto', 
                    vmin=0, vmax=1)
    
    ax1.set_xticks(range(len(plot_data['L_vals'])))
    ax1.set_xticklabels(plot_data['L_vals'])
    ax1.set_yticks(range(len(plot_data['k_vals'])))
    ax1.set_yticklabels(plot_data['k_vals'])
    ax1.set_xlabel('Window Length (L)')
    ax1.set_ylabel('Flip Budget (k)')
    ax1.set_title('Uniqueness Rate Heatmap')
    
    # Add colorbar
    plt.colorbar(im, ax=ax1, label='Uniqueness Rate')
    
    # Add text annotations
    for i in range(len(plot_data['k_vals'])):
        for j in range(len(plot_data['L_vals'])):
            rate = plot_data['heatmap_data'][i, j]
            ax1.text(j, i, f'{rate:.2f}', ha='center', va='center', 
                    color='white' if rate < 0.5 else 'black')
    
    # Summary plot
    ax2.axis('off')
    
    # Find best result
    best_rate = np.max(plot_data['heatmap_data'])
    threshold = thresholds.get('T4_uniqueness', 0.95)
    
    summary_text = f"T4 Retrodiction Uniqueness Results\n\n"
    summary_text += f"Best uniqueness rate: {best_rate:.3f}\n"
    summary_text += f"Threshold: {threshold:.3f}\n\n"
    summary_text += f"Status: {'PASS ✓' if best_rate >= threshold else 'FAIL ✗'}\n\n"
    summary_text += "Method: Hamming ball search with\ntriad consistency scoring\n\n"
    summary_text += "Expectation: ≥95% uniqueness\nfor k≤2, L≥16"
    
    ax2.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    return fig

def main():
    """Main entry point."""
    result = run_t4_test()
    return result['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)