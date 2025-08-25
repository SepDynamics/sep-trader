#!/usr/bin/env python3
"""
Test T3: Convolution Invariance
Tests that antialiased decimation + time rescaling preserves triad shape for band-limited signals

Uses D2 mapping (scale-invariant)
Shows failure without antialiasing as negative control
"""

import sys
import os
# Add the parent directory (validation) to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, List, Tuple

# Import from shared utilities
from common import (
    compute_triad,
    mapping_D2_dilation_robust,
    compute_joint_rmse,
    apply_antialiasing_filter,
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
from plots import plot_t3_results, setup_plot_style
from thresholds import (
    get_thresholds,
    validate_t3_results,
    get_hypothesis_description
)

# Test parameters
SIGNAL_LENGTH = 50000  # Length of test signal
BETA = 0.1  # EMA parameter for triad computation
DECIMATION_FACTORS = [2, 4]  # Decimation factors to test
SEEDS = [1337, 1729, 2718]  # Random seeds
CHIRP_F0 = 0.01  # Starting frequency of chirp
CHIRP_F1 = 0.1   # Ending frequency of chirp
NOISE_LEVEL = 0.1  # Noise level (20 dB SNR)

def generate_chirp_signal(length: int, f0: float, f1: float, 
                         noise_level: float, seed: int) -> np.ndarray:
    """Generate a chirp signal with noise."""
    set_random_seed(seed)
    
    # Time vector
    t = np.linspace(0, 1, length)
    
    # Linear chirp frequency
    freq = f0 + (f1 - f0) * t
    phase = 2 * np.pi * np.cumsum(freq) / length
    
    # Generate chirp
    chirp = np.sin(phase)
    
    # Add noise
    noise = np.random.randn(length) * noise_level
    signal_with_noise = chirp + noise
    
    # Convert to price-like signal (always positive)
    signal_with_noise = signal_with_noise - np.min(signal_with_noise) + 1.0
    
    return signal_with_noise

def decimate_signal(input_signal: np.ndarray, factor: int, use_antialiasing: bool) -> np.ndarray:
    """Decimate signal with or without antialiasing."""
    if use_antialiasing:
        # Apply antialiasing filter
        # Cutoff frequency should be 1/(2*factor) to satisfy Nyquist
        cutoff = 0.5 / factor
        
        # Design Butterworth filter
        b, a = signal.butter(8, cutoff, btype='low')
        filtered = signal.filtfilt(b, a, input_signal)
        
        # Decimate
        decimated = filtered[::factor]
    else:
        # Direct decimation without filtering (aliasing will occur)
        decimated = input_signal[::factor]
    
    return decimated

def time_rescale_to_match(decimated: np.ndarray, original_length: int, 
                         decimation_factor: int) -> np.ndarray:
    """Rescale decimated signal back to original time scale."""
    # The decimated signal represents the same time duration but with fewer samples
    # We need to interpolate it back to the original sample rate
    
    decimated_length = len(decimated)
    
    # Create time axes
    t_decimated = np.linspace(0, 1, decimated_length)
    t_original = np.linspace(0, 1, original_length)
    
    # Interpolate to original time grid
    rescaled = np.interp(t_original, t_decimated, decimated)
    
    return rescaled

def run_single_decimation_test(signal: np.ndarray, decimation_factor: int,
                               use_antialiasing: bool, seed: int) -> Dict:
    """Run a single decimation test."""
    # Apply D2 mapping to original signal
    chords_orig = mapping_D2_dilation_robust(signal)
    triads_orig = compute_triad(chords_orig, beta=BETA)
    
    # Decimate signal
    decimated = decimate_signal(signal, decimation_factor, use_antialiasing)
    
    # Time-rescale back to original length
    rescaled = time_rescale_to_match(decimated, len(signal), decimation_factor)
    
    # Apply D2 mapping to rescaled signal
    chords_rescaled = mapping_D2_dilation_robust(rescaled)
    triads_rescaled = compute_triad(chords_rescaled, beta=BETA)
    
    # Compute RMSE
    joint_rmse = compute_joint_rmse(triads_orig, triads_rescaled)
    
    return {
        'decimation_factor': decimation_factor,
        'use_antialiasing': use_antialiasing,
        'seed': seed,
        'joint_rmse': joint_rmse,
        'triads_orig': triads_orig,
        'triads_rescaled': triads_rescaled
    }

def run_t3_test() -> Dict:
    """Run the complete T3 test suite."""
    
    with TestLogger("T3", "Convolution Invariance"):
        results_with_aa = []
        results_without_aa = []
        
        # Test each configuration
        for seed in SEEDS:
            print(f"  Testing with seed={seed}")
            
            # Generate test signal
            signal = generate_chirp_signal(SIGNAL_LENGTH, CHIRP_F0, CHIRP_F1, 
                                          NOISE_LEVEL, seed)
            
            for decimation_factor in DECIMATION_FACTORS:
                # Test with antialiasing
                print(f"    Decimation factor {decimation_factor} with antialiasing")
                result_aa = run_single_decimation_test(signal, decimation_factor, 
                                                       True, seed)
                results_with_aa.append(result_aa)
                print(f"      RMSE: {result_aa['joint_rmse']:.4f}")
                
                # Test without antialiasing (negative control)
                print(f"    Decimation factor {decimation_factor} without antialiasing")
                result_no_aa = run_single_decimation_test(signal, decimation_factor,
                                                          False, seed)
                results_without_aa.append(result_no_aa)
                print(f"      RMSE: {result_no_aa['joint_rmse']:.4f}")
        
        # Aggregate results by decimation factor
        rmse_by_decimation_aa = {}
        rmse_by_decimation_no_aa = {}
        
        for factor in DECIMATION_FACTORS:
            # With antialiasing
            factor_results_aa = [r['joint_rmse'] for r in results_with_aa 
                                 if r['decimation_factor'] == factor]
            rmse_by_decimation_aa[factor] = np.median(factor_results_aa)
            
            # Without antialiasing
            factor_results_no_aa = [r['joint_rmse'] for r in results_without_aa
                                    if r['decimation_factor'] == factor]
            rmse_by_decimation_no_aa[factor] = np.median(factor_results_no_aa)
        
        # Overall medians
        median_rmse_aa = np.median([r['joint_rmse'] for r in results_with_aa])
        median_rmse_no_aa = np.median([r['joint_rmse'] for r in results_without_aa])
        
        # Validate results
        validation = validate_t3_results(median_rmse_aa, median_rmse_no_aa)
        
        # Get thresholds
        thresholds = get_thresholds('T3')
        
        # Log hypothesis results
        log_hypothesis("T3", get_hypothesis_description('T3', 'T3'),
                      thresholds['T3'], median_rmse_aa, validation['T3'])
        
        # Log control results
        print("\nNegative Control (No Antialiasing):")
        print(f"  Median RMSE: {median_rmse_no_aa:.4f} (should be > {thresholds['T3']:.3f})")
        print(f"  Control shows aliasing artifacts: {validation['control_fail']}")
        print(f"  RMSE ratio (no AA / with AA): {validation['rmse_ratio']:.2f}")
        
        # Get example triads for plotting
        example_orig = results_with_aa[0]['triads_orig']
        example_decimated = results_with_aa[0]['triads_rescaled']
        
        # Prepare summary
        summary = {
            'test': 'T3',
            'parameters': {
                'signal_length': SIGNAL_LENGTH,
                'beta': BETA,
                'decimation_factors': DECIMATION_FACTORS,
                'seeds': SEEDS,
                'chirp_f0': CHIRP_F0,
                'chirp_f1': CHIRP_F1,
                'noise_level': NOISE_LEVEL,
                'mapping': 'D2'
            },
            'results': {
                'with_antialiasing': {
                    'median_rmse': median_rmse_aa,
                    'rmse_by_decimation': rmse_by_decimation_aa
                },
                'without_antialiasing': {
                    'median_rmse': median_rmse_no_aa,
                    'rmse_by_decimation': rmse_by_decimation_no_aa
                },
                'rmse_ratio': validation['rmse_ratio']
            },
            'hypothesis': {
                'T3': {
                    'pass': validation['T3'],
                    'metric': median_rmse_aa,
                    'threshold': thresholds['T3'],
                    'description': get_hypothesis_description('T3', 'T3')
                }
            },
            'control': {
                'failed_as_expected': validation['control_fail'],
                'description': 'Decimation without antialiasing should produce high RMSE'
            },
            'overall_pass': validation['T3']
        }
        
        # Prepare plot data
        plot_data = {
            'decimations': DECIMATION_FACTORS,
            'rmse_antialiased': [rmse_by_decimation_aa[d] for d in DECIMATION_FACTORS],
            'rmse_no_antialiasing': [rmse_by_decimation_no_aa[d] for d in DECIMATION_FACTORS],
            'original_triad': example_orig,
            'decimated_triad': example_decimated
        }
        
        # Create plot
        fig = plot_t3_results(plot_data, thresholds)
        
        # Prepare metrics for CSV
        metrics = []
        for r in results_with_aa:
            metrics.append({
                'decimation_factor': r['decimation_factor'],
                'antialiasing': 'yes',
                'seed': r['seed'],
                'joint_rmse': r['joint_rmse']
            })
        for r in results_without_aa:
            metrics.append({
                'decimation_factor': r['decimation_factor'],
                'antialiasing': 'no',
                'seed': r['seed'],
                'joint_rmse': r['joint_rmse']
            })
        
        # Save results
        summary_clean = format_json_safe(summary)
        save_test_results('T3', summary_clean, metrics, fig)
        
        # Log summary
        log_test_summary('T3', {'T3': validation['T3']})
        
        # Show decimation-specific results
        print("\nResults by decimation factor:")
        for factor in DECIMATION_FACTORS:
            print(f"  Factor {factor}:")
            print(f"    With AA:    {rmse_by_decimation_aa[factor]:.4f}")
            print(f"    Without AA: {rmse_by_decimation_no_aa[factor]:.4f}")
        
        return summary

def main():
    """Main entry point."""
    result = run_t3_test()
    return result['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)