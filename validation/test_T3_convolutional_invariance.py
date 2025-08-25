#!/usr/bin/env python3
"""
Test T3: Convolutional Invariance on Band-Limited Waves
Tests H5: Triad metrics remain stable under controlled decimation of band-limited waves
Tests H6: Convolutional operations preserve triad structure for appropriately band-limited signals
"""

import numpy as np
import json
import csv
import os
from pathlib import Path
from scipy import signal
from sep_core import (
    triad_series, rmse, bit_mapping_D1, bit_mapping_D2, bit_mapping_D3,
    generate_chirp, controlled_convolution_decimation, RANDOM_SEED
)
import matplotlib.pyplot as plt

# Test parameters
PROCESS_LENGTH = 100000
BETA = 0.1  # EMA parameter
SAMPLING_RATES = [1000, 500, 250]  # Hz (decimation factors 1, 2, 4)
CUTOFF_FREQUENCIES = [50, 100, 200]  # Hz (relative to Nyquist)
DECIMATION_FACTORS = [1, 2, 4, 8]
RMSE_THRESHOLD_H5 = 0.1  # Max RMSE for band-limited invariance
RMSE_THRESHOLD_H6 = 0.15  # Max RMSE for convolutional invariance

def create_results_dir():
    """Create results directory if it doesn't exist."""
    Path("results").mkdir(exist_ok=True)

def generate_band_limited_wave(length: int, fs: int, fc: float, wave_type: str = "chirp", seed: int = RANDOM_SEED) -> np.ndarray:
    """Generate band-limited wave with specified cutoff frequency."""
    np.random.seed(seed)
    
    if wave_type == "chirp":
        # Generate linear chirp from fc/4 to fc
        t = np.arange(length) / fs
        f0, f1 = fc/4, fc
        wave = generate_chirp(length, f0=f0, f1=f1, fs=fs, seed=seed)
    elif wave_type == "filtered_noise":
        # Generate white noise and band-limit it
        noise = np.random.normal(0, 1, length)
        # Design low-pass filter
        nyquist = fs / 2
        normalized_cutoff = fc / nyquist
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        wave = signal.filtfilt(b, a, noise)
    elif wave_type == "multitone":
        # Generate sum of sinusoids below cutoff
        t = np.arange(length) / fs
        wave = np.zeros(length)
        n_tones = 5
        freqs = np.linspace(fc/10, fc*0.8, n_tones)
        for f in freqs:
            phase = np.random.uniform(0, 2*np.pi)
            wave += np.sin(2*np.pi*f*t + phase)
        wave /= np.std(wave)  # Normalize
    else:
        raise ValueError(f"Unknown wave type: {wave_type}")
    
    return wave

def apply_decimation(signal_data: np.ndarray, decimation_factor: int) -> np.ndarray:
    """Apply decimation with anti-aliasing filter."""
    if decimation_factor == 1:
        return signal_data
    
    # Apply anti-aliasing filter before decimation
    # Cutoff at 0.8 * Nyquist frequency of decimated signal
    cutoff = 0.8 / decimation_factor
    b, a = signal.butter(4, cutoff, btype='low')
    filtered = signal.filtfilt(b, a, signal_data)
    
    # Decimate
    decimated = filtered[::decimation_factor]
    
    return decimated

def apply_convolution_then_decimation(signal_data: np.ndarray, decimation_factor: int, 
                                    kernel_type: str = "gaussian") -> np.ndarray:
    """Apply convolution followed by decimation as controlled operation."""
    if decimation_factor == 1:
        return signal_data
    
    # Generate convolution kernel
    kernel_size = min(2 * decimation_factor + 1, 21)  # Odd size
    if kernel_type == "gaussian":
        sigma = decimation_factor / 3.0  # Scale sigma with decimation
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel /= np.sum(kernel)
    elif kernel_type == "box":
        kernel = np.ones(kernel_size) / kernel_size
    elif kernel_type == "triangle":
        center = kernel_size // 2
        kernel = np.maximum(0, 1 - np.abs(np.arange(kernel_size) - center) / center)
        kernel /= np.sum(kernel)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Apply convolution
    convolved = np.convolve(signal_data, kernel, mode='same')
    
    # Decimate
    decimated = convolved[::decimation_factor]
    
    return decimated

def align_decimated_triads(original_triads: np.ndarray, decimated_triads: np.ndarray, 
                          decimation_factor: int) -> np.ndarray:
    """Align decimated triads with original by upsampling."""
    if decimation_factor == 1:
        return decimated_triads
    
    # Upsample decimated triads to match original length
    target_length = len(original_triads)
    current_length = len(decimated_triads)
    
    # Create time vectors
    original_time = np.linspace(0, 1, target_length)
    decimated_time = np.linspace(0, 1, current_length)
    
    # Interpolate each triad component
    aligned_triads = np.zeros((target_length, 3))
    for i in range(3):  # H, C, S
        aligned_triads[:, i] = np.interp(original_time, decimated_time, 
                                       decimated_triads[:, i])
    
    return aligned_triads

def compute_triad_stability_metric(triads: np.ndarray) -> dict:
    """Compute stability metrics for triad series."""
    H, C, S = triads[:, 0], triads[:, 1], triads[:, 2]
    
    # Compute variance of each component
    H_var = np.var(H)
    C_var = np.var(C)
    S_var = np.var(S)
    
    # Compute trend (linear fit slope)
    t = np.arange(len(triads))
    H_trend = np.polyfit(t, H, 1)[0]
    C_trend = np.polyfit(t, C, 1)[0]
    S_trend = np.polyfit(t, S, 1)[0]
    
    # Compute autocorrelation at lag 1
    def autocorr_lag1(x):
        if len(x) < 2:
            return 0.0
        x_centered = x - np.mean(x)
        return np.corrcoef(x_centered[:-1], x_centered[1:])[0, 1]
    
    H_autocorr = autocorr_lag1(H)
    C_autocorr = autocorr_lag1(C)
    S_autocorr = autocorr_lag1(S)
    
    return {
        'variance': {'H': H_var, 'C': C_var, 'S': S_var},
        'trend': {'H': H_trend, 'C': C_trend, 'S': S_trend},
        'autocorr': {'H': H_autocorr, 'C': C_autocorr, 'S': S_autocorr}
    }

def run_decimation_invariance_test(wave_type: str, fs: int, fc: float, bit_mapping: str, 
                                 seed: int = RANDOM_SEED) -> dict:
    """Run decimation invariance test for given parameters."""
    print(f"Running T3 decimation test: {wave_type} wave, fs={fs}Hz, fc={fc}Hz, {bit_mapping} mapping")
    
    # Generate band-limited wave
    wave = generate_band_limited_wave(PROCESS_LENGTH, fs, fc, wave_type, seed)
    
    # Apply bit mapping to original
    if bit_mapping == "D1":
        bits_orig = bit_mapping_D1(wave)
    elif bit_mapping == "D2":
        bits_orig = bit_mapping_D2(wave)
    elif bit_mapping == "D3":
        bits_orig = bit_mapping_D3(wave)
    else:
        raise ValueError(f"Unknown bit mapping: {bit_mapping}")
    
    # Compute original triads
    triads_orig = triad_series(bits_orig, beta=BETA)
    orig_stability = compute_triad_stability_metric(triads_orig)
    
    results = {
        'wave_type': wave_type,
        'fs': fs,
        'fc': fc,
        'bit_mapping': bit_mapping,
        'seed': seed,
        'original_stability': orig_stability,
        'decimation_results': {}
    }
    
    # Test each decimation factor
    for dec_factor in DECIMATION_FACTORS:
        if dec_factor == 1:
            # Skip decimation factor 1 (already have original)
            continue
            
        print(f"  Testing decimation factor {dec_factor}")
        
        # Check if decimation is theoretically valid (Nyquist criterion)
        effective_fs = fs / dec_factor
        nyquist_after_decimation = effective_fs / 2
        theoretically_valid = fc <= nyquist_after_decimation * 0.8  # 80% safety margin
        
        # Apply decimation
        decimated_wave = apply_decimation(wave, dec_factor)
        
        # Apply same bit mapping to decimated wave
        if bit_mapping == "D1":
            bits_dec = bit_mapping_D1(decimated_wave)
        elif bit_mapping == "D2":
            bits_dec = bit_mapping_D2(decimated_wave)
        elif bit_mapping == "D3":
            bits_dec = bit_mapping_D3(decimated_wave)
        
        # Compute decimated triads
        triads_dec = triad_series(bits_dec, beta=BETA)
        
        # Align with original
        triads_aligned = align_decimated_triads(triads_orig, triads_dec, dec_factor)
        
        # Compute metrics
        joint_rmse = rmse(triads_orig.flatten(), triads_aligned.flatten())
        h_rmse = rmse(triads_orig[:, 0], triads_aligned[:, 0])
        c_rmse = rmse(triads_orig[:, 1], triads_aligned[:, 1])
        s_rmse = rmse(triads_orig[:, 2], triads_aligned[:, 2])
        
        dec_stability = compute_triad_stability_metric(triads_aligned)
        
        results['decimation_results'][dec_factor] = {
            'theoretically_valid': theoretically_valid,
            'effective_fs': effective_fs,
            'nyquist_after_decimation': nyquist_after_decimation,
            'joint_rmse': joint_rmse,
            'component_rmse': {'H': h_rmse, 'C': c_rmse, 'S': s_rmse},
            'decimated_stability': dec_stability
        }
        
        print(f"    Theoretically valid: {theoretically_valid}")
        print(f"    Joint RMSE: {joint_rmse:.6f}")
        print(f"    Component RMSEs - H: {h_rmse:.6f}, C: {c_rmse:.6f}, S: {s_rmse:.6f}")
    
    return results

def run_convolution_invariance_test(wave_type: str, fs: int, fc: float, bit_mapping: str,
                                   kernel_type: str = "gaussian", seed: int = RANDOM_SEED) -> dict:
    """Run convolution + decimation invariance test."""
    print(f"Running T3 convolution test: {wave_type} wave, {kernel_type} kernel, {bit_mapping} mapping")
    
    # Generate band-limited wave
    wave = generate_band_limited_wave(PROCESS_LENGTH, fs, fc, wave_type, seed)
    
    # Apply bit mapping to original
    if bit_mapping == "D1":
        bits_orig = bit_mapping_D1(wave)
    elif bit_mapping == "D2":
        bits_orig = bit_mapping_D2(wave)
    elif bit_mapping == "D3":
        bits_orig = bit_mapping_D3(wave)
    else:
        raise ValueError(f"Unknown bit mapping: {bit_mapping}")
    
    # Compute original triads
    triads_orig = triad_series(bits_orig, beta=BETA)
    
    results = {
        'wave_type': wave_type,
        'fs': fs,
        'fc': fc,
        'bit_mapping': bit_mapping,
        'kernel_type': kernel_type,
        'seed': seed,
        'convolution_results': {}
    }
    
    # Test each decimation factor with convolution
    for dec_factor in DECIMATION_FACTORS:
        if dec_factor == 1:
            continue
            
        print(f"  Testing convolution + decimation factor {dec_factor}")
        
        # Apply convolution then decimation
        conv_dec_wave = apply_convolution_then_decimation(wave, dec_factor, kernel_type)
        
        # Apply same bit mapping
        if bit_mapping == "D1":
            bits_conv_dec = bit_mapping_D1(conv_dec_wave)
        elif bit_mapping == "D2":
            bits_conv_dec = bit_mapping_D2(conv_dec_wave)
        elif bit_mapping == "D3":
            bits_conv_dec = bit_mapping_D3(conv_dec_wave)
        
        # Compute triads
        triads_conv_dec = triad_series(bits_conv_dec, beta=BETA)
        
        # Align with original
        triads_aligned = align_decimated_triads(triads_orig, triads_conv_dec, dec_factor)
        
        # Compute metrics
        joint_rmse = rmse(triads_orig.flatten(), triads_aligned.flatten())
        h_rmse = rmse(triads_orig[:, 0], triads_aligned[:, 0])
        c_rmse = rmse(triads_orig[:, 1], triads_aligned[:, 1])
        s_rmse = rmse(triads_orig[:, 2], triads_aligned[:, 2])
        
        results['convolution_results'][dec_factor] = {
            'joint_rmse': joint_rmse,
            'component_rmse': {'H': h_rmse, 'C': c_rmse, 'S': s_rmse}
        }
        
        print(f"    Joint RMSE: {joint_rmse:.6f}")
    
    return results

def evaluate_hypotheses(decimation_results: list, convolution_results: list) -> dict:
    """Evaluate H5 and H6 hypotheses."""
    
    # H5: Band-limited decimation preserves triad structure
    valid_decimation_rmses = []
    invalid_decimation_rmses = []
    
    for res in decimation_results:
        for dec_factor, dec_data in res['decimation_results'].items():
            rmse_val = dec_data['joint_rmse']
            if dec_data['theoretically_valid']:
                valid_decimation_rmses.append(rmse_val)
            else:
                invalid_decimation_rmses.append(rmse_val)
    
    median_valid_rmse = np.median(valid_decimation_rmses) if valid_decimation_rmses else float('inf')
    median_invalid_rmse = np.median(invalid_decimation_rmses) if invalid_decimation_rmses else float('inf')
    
    h5_pass = median_valid_rmse <= RMSE_THRESHOLD_H5
    
    # H6: Convolution + decimation preserves triad structure
    convolution_rmses = []
    
    for res in convolution_results:
        for dec_factor, conv_data in res['convolution_results'].items():
            convolution_rmses.append(conv_data['joint_rmse'])
    
    median_convolution_rmse = np.median(convolution_rmses) if convolution_rmses else float('inf')
    h6_pass = median_convolution_rmse <= RMSE_THRESHOLD_H6
    
    evaluation = {
        'H5_band_limited_decimation': {
            'median_valid_rmse': median_valid_rmse,
            'median_invalid_rmse': median_invalid_rmse,
            'threshold': RMSE_THRESHOLD_H5,
            'n_valid_tests': len(valid_decimation_rmses),
            'n_invalid_tests': len(invalid_decimation_rmses),
            'pass': h5_pass
        },
        'H6_convolution_invariance': {
            'median_rmse': median_convolution_rmse,
            'threshold': RMSE_THRESHOLD_H6,
            'n_tests': len(convolution_rmses),
            'pass': h6_pass
        },
        'overall_pass': h5_pass and h6_pass
    }
    
    return evaluation

def save_results(decimation_results: list, convolution_results: list, evaluation: dict):
    """Save results to files."""
    create_results_dir()
    
    # Save detailed results to JSON
    output_data = {
        'test': 'T3_convolutional_invariance',
        'parameters': {
            'process_length': PROCESS_LENGTH,
            'beta': BETA,
            'sampling_rates': SAMPLING_RATES,
            'cutoff_frequencies': CUTOFF_FREQUENCIES,
            'decimation_factors': DECIMATION_FACTORS,
            'rmse_threshold_h5': RMSE_THRESHOLD_H5,
            'rmse_threshold_h6': RMSE_THRESHOLD_H6
        },
        'decimation_results': decimation_results,
        'convolution_results': convolution_results,
        'evaluation': evaluation
    }
    
    with open('results/T3_summary.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save metrics to CSV
    with open('results/T3_decimation_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['wave_type', 'fs', 'fc', 'bit_mapping', 'decimation_factor',
                        'theoretically_valid', 'joint_rmse', 'H_rmse', 'C_rmse', 'S_rmse'])
        
        for res in decimation_results:
            for dec_factor, dec_data in res['decimation_results'].items():
                writer.writerow([
                    res['wave_type'], res['fs'], res['fc'], res['bit_mapping'],
                    dec_factor, dec_data['theoretically_valid'],
                    dec_data['joint_rmse'],
                    dec_data['component_rmse']['H'],
                    dec_data['component_rmse']['C'],
                    dec_data['component_rmse']['S']
                ])

def create_plots(decimation_results: list, convolution_results: list, evaluation: dict):
    """Create visualization plots."""
    create_results_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('T3: Convolutional Invariance Results')
    
    # Plot H5: Decimation RMSE by validity
    ax = axes[0, 0]
    valid_rmses = []
    invalid_rmses = []
    dec_factors = []
    
    for res in decimation_results:
        for dec_factor, dec_data in res['decimation_results'].items():
            dec_factors.append(dec_factor)
            if dec_data['theoretically_valid']:
                valid_rmses.append(dec_data['joint_rmse'])
                invalid_rmses.append(np.nan)
            else:
                valid_rmses.append(np.nan)
                invalid_rmses.append(dec_data['joint_rmse'])
    
    # Plot by decimation factor
    unique_factors = sorted(set(dec_factors))
    valid_by_factor = {f: [] for f in unique_factors}
    invalid_by_factor = {f: [] for f in unique_factors}
    
    for i, dec_factor in enumerate(dec_factors):
        if not np.isnan(valid_rmses[i]):
            valid_by_factor[dec_factor].append(valid_rmses[i])
        if not np.isnan(invalid_rmses[i]):
            invalid_by_factor[dec_factor].append(invalid_rmses[i])
    
    for factor in unique_factors:
        if valid_by_factor[factor]:
            ax.scatter([factor] * len(valid_by_factor[factor]), 
                      valid_by_factor[factor], c='green', alpha=0.7, s=50, label='Valid' if factor == unique_factors[0] else "")
        if invalid_by_factor[factor]:
            ax.scatter([factor] * len(invalid_by_factor[factor]), 
                      invalid_by_factor[factor], c='red', alpha=0.7, s=50, label='Invalid' if factor == unique_factors[0] else "")
    
    ax.axhline(y=RMSE_THRESHOLD_H5, color='k', linestyle='--', alpha=0.5, label=f'H5 Threshold ({RMSE_THRESHOLD_H5})')
    ax.set_xlabel('Decimation Factor')
    ax.set_ylabel('Joint RMSE')
    ax.set_title('H5: Band-Limited Decimation Invariance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot H6: Convolution RMSE
    ax = axes[0, 1]
    conv_rmses = []
    conv_factors = []
    
    for res in convolution_results:
        for dec_factor, conv_data in res['convolution_results'].items():
            conv_factors.append(dec_factor)
            conv_rmses.append(conv_data['joint_rmse'])
    
    # Group by decimation factor
    conv_by_factor = {f: [] for f in unique_factors}
    for i, factor in enumerate(conv_factors):
        conv_by_factor[factor].append(conv_rmses[i])
    
    box_data = [conv_by_factor[f] for f in unique_factors if conv_by_factor[f]]
    box_labels = [str(f) for f in unique_factors if conv_by_factor[f]]
    
    if box_data:
        ax.boxplot(box_data, labels=box_labels)
    ax.axhline(y=RMSE_THRESHOLD_H6, color='r', linestyle='--', alpha=0.7, 
               label=f'H6 Threshold ({RMSE_THRESHOLD_H6})')
    ax.set_xlabel('Decimation Factor')
    ax.set_ylabel('Joint RMSE')
    ax.set_title('H6: Convolution + Decimation Invariance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Component-wise RMSE comparison
    ax = axes[1, 0]
    components = ['H', 'C', 'S']
    valid_comp_rmses = {comp: [] for comp in components}
    invalid_comp_rmses = {comp: [] for comp in components}
    
    for res in decimation_results:
        for dec_factor, dec_data in res['decimation_results'].items():
            for comp in components:
                if dec_data['theoretically_valid']:
                    valid_comp_rmses[comp].append(dec_data['component_rmse'][comp])
                else:
                    invalid_comp_rmses[comp].append(dec_data['component_rmse'][comp])
    
    x_pos = np.arange(len(components))
    width = 0.35
    
    valid_means = [np.mean(valid_comp_rmses[comp]) if valid_comp_rmses[comp] else 0 for comp in components]
    invalid_means = [np.mean(invalid_comp_rmses[comp]) if invalid_comp_rmses[comp] else 0 for comp in components]
    
    ax.bar(x_pos - width/2, valid_means, width, label='Valid Decimation', alpha=0.7, color='green')
    ax.bar(x_pos + width/2, invalid_means, width, label='Invalid Decimation', alpha=0.7, color='red')
    
    ax.set_xlabel('Triad Component')
    ax.set_ylabel('Mean RMSE')
    ax.set_title('Component-wise RMSE: Valid vs Invalid Decimation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(components)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Evaluation summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create evaluation summary text
    summary_text = f"""
Evaluation Summary:

H5 (Band-Limited Decimation):
  Status: {'PASS' if evaluation['H5_band_limited_decimation']['pass'] else 'FAIL'}
  Valid Tests Median RMSE: {evaluation['H5_band_limited_decimation']['median_valid_rmse']:.4f}
  Threshold: {evaluation['H5_band_limited_decimation']['threshold']:.4f}
  Valid Tests: {evaluation['H5_band_limited_decimation']['n_valid_tests']}
  Invalid Tests: {evaluation['H5_band_limited_decimation']['n_invalid_tests']}

H6 (Convolution Invariance):
  Status: {'PASS' if evaluation['H6_convolution_invariance']['pass'] else 'FAIL'}
  Median RMSE: {evaluation['H6_convolution_invariance']['median_rmse']:.4f}
  Threshold: {evaluation['H6_convolution_invariance']['threshold']:.4f}
  Total Tests: {evaluation['H6_convolution_invariance']['n_tests']}

Overall Test: {'PASS' if evaluation['overall_pass'] else 'FAIL'}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/T3_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the complete T3 test suite."""
    print("="*60)
    print("Running Test T3: Convolutional Invariance on Band-Limited Waves")
    print("="*60)
    
    decimation_results = []
    convolution_results = []
    
    # Test decimation invariance
    print("\n--- Testing Decimation Invariance (H5) ---")
    for wave_type in ["chirp", "filtered_noise"]:
        for fs in [1000, 500]:
            for fc in [50, 100]:
                if fc >= fs/2:  # Skip invalid combinations
                    continue
                for bit_mapping in ["D1", "D2"]:
                    result = run_decimation_invariance_test(wave_type, fs, fc, bit_mapping, RANDOM_SEED)
                    decimation_results.append(result)
    
    # Test convolution invariance
    print("\n--- Testing Convolution Invariance (H6) ---")
    for wave_type in ["chirp", "multitone"]:
        for kernel_type in ["gaussian", "box"]:
            for bit_mapping in ["D1", "D2"]:
                result = run_convolution_invariance_test(wave_type, 1000, 100, bit_mapping, 
                                                       kernel_type, RANDOM_SEED + 1)
                convolution_results.append(result)
    
    # Evaluate hypotheses
    evaluation = evaluate_hypotheses(decimation_results, convolution_results)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"H5 (Band-Limited Decimation): {'PASS' if evaluation['H5_band_limited_decimation']['pass'] else 'FAIL'}")
    print(f"  Valid tests median RMSE: {evaluation['H5_band_limited_decimation']['median_valid_rmse']:.6f}")
    print(f"  Invalid tests median RMSE: {evaluation['H5_band_limited_decimation']['median_invalid_rmse']:.6f}")
    print(f"  Threshold: {evaluation['H5_band_limited_decimation']['threshold']:.6f}")
    print(f"  Valid tests count: {evaluation['H5_band_limited_decimation']['n_valid_tests']}")
    print(f"  Invalid tests count: {evaluation['H5_band_limited_decimation']['n_invalid_tests']}")
    
    print(f"\nH6 (Convolution Invariance): {'PASS' if evaluation['H6_convolution_invariance']['pass'] else 'FAIL'}")
    print(f"  Median RMSE: {evaluation['H6_convolution_invariance']['median_rmse']:.6f}")
    print(f"  Threshold: {evaluation['H6_convolution_invariance']['threshold']:.6f}")
    print(f"  Test count: {evaluation['H6_convolution_invariance']['n_tests']}")
    
    print(f"\nOVERALL TEST: {'PASS' if evaluation['overall_pass'] else 'FAIL'}")
    
    # Save results
    save_results(decimation_results, convolution_results, evaluation)
    create_plots(decimation_results, convolution_results, evaluation)
    
    print(f"\nResults saved to results/T3_summary.json and results/T3_decimation_metrics.csv")
    print(f"Plots saved to results/T3_plots.png")
    
    return evaluation['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)