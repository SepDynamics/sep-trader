#!/usr/bin/env python3
"""
Test T4: Retrodictive Reconstruction With Continuity Constraint
Tests H7: Retrodictive reconstruction using triad-informed priors outperforms naive interpolation
Tests H8: Continuity constraints improve reconstruction accuracy for smooth underlying processes
"""

import numpy as np
import json
import csv
import os
from pathlib import Path
from scipy import interpolate, optimize
from scipy.ndimage import gaussian_filter1d
from sep_core import (
    triad_series, rmse, bit_mapping_D1, bit_mapping_D2,
    generate_poisson_process, generate_van_der_pol, generate_chirp, RANDOM_SEED
)
import matplotlib.pyplot as plt

# Test parameters
PROCESS_LENGTH = 50000
BETA = 0.1  # EMA parameter
GAP_SIZES = [100, 200, 500, 1000]  # Missing data gap sizes
GAP_POSITIONS = ["beginning", "middle", "end"]  # Where to place gaps
RECONSTRUCTION_METHODS = ["linear", "cubic", "triad_informed", "constrained"]
IMPROVEMENT_THRESHOLD_H7 = 0.2  # Min improvement ratio for triad-informed vs naive
IMPROVEMENT_THRESHOLD_H8 = 0.15  # Min improvement ratio for continuity constraints

def create_results_dir():
    """Create results directory if it doesn't exist."""
    Path("results").mkdir(exist_ok=True)

def create_missing_data_gap(signal: np.ndarray, gap_size: int, position: str = "middle") -> tuple:
    """Create gap in signal data and return indices of missing data."""
    signal_copy = signal.copy()
    n = len(signal)
    
    if position == "beginning":
        start_idx = 0
        end_idx = min(gap_size, n)
    elif position == "end":
        start_idx = max(0, n - gap_size)
        end_idx = n
    else:  # middle
        center = n // 2
        start_idx = max(0, center - gap_size // 2)
        end_idx = min(n, start_idx + gap_size)
    
    # Create mask for missing data
    missing_mask = np.zeros(n, dtype=bool)
    missing_mask[start_idx:end_idx] = True
    
    # Set missing data to NaN
    signal_copy[missing_mask] = np.nan
    
    return signal_copy, missing_mask, np.arange(start_idx, end_idx)

def linear_interpolation(signal_with_gaps: np.ndarray, missing_indices: np.ndarray) -> np.ndarray:
    """Perform linear interpolation to fill gaps."""
    signal_reconstructed = signal_with_gaps.copy()
    valid_mask = ~np.isnan(signal_with_gaps)
    
    if np.sum(valid_mask) < 2:
        return signal_reconstructed  # Cannot interpolate with less than 2 points
    
    valid_indices = np.where(valid_mask)[0]
    valid_values = signal_with_gaps[valid_mask]
    
    # Linear interpolation
    interpolator = interpolate.interp1d(valid_indices, valid_values, 
                                       kind='linear', fill_value='extrapolate')
    signal_reconstructed[missing_indices] = interpolator(missing_indices)
    
    return signal_reconstructed

def cubic_interpolation(signal_with_gaps: np.ndarray, missing_indices: np.ndarray) -> np.ndarray:
    """Perform cubic spline interpolation to fill gaps."""
    signal_reconstructed = signal_with_gaps.copy()
    valid_mask = ~np.isnan(signal_with_gaps)
    
    if np.sum(valid_mask) < 4:
        # Fall back to linear if insufficient points for cubic
        return linear_interpolation(signal_with_gaps, missing_indices)
    
    valid_indices = np.where(valid_mask)[0]
    valid_values = signal_with_gaps[valid_mask]
    
    # Cubic spline interpolation
    interpolator = interpolate.interp1d(valid_indices, valid_values, 
                                       kind='cubic', fill_value='extrapolate')
    signal_reconstructed[missing_indices] = interpolator(missing_indices)
    
    return signal_reconstructed

def triad_informed_reconstruction(signal_with_gaps: np.ndarray, missing_indices: np.ndarray, 
                                bit_mapping: str) -> np.ndarray:
    """Reconstruction using triad-informed priors."""
    signal_reconstructed = signal_with_gaps.copy()
    valid_mask = ~np.isnan(signal_with_gaps)
    
    if np.sum(valid_mask) < 10:
        # Fall back to cubic if insufficient data
        return cubic_interpolation(signal_with_gaps, missing_indices)
    
    # Get valid signal portions
    valid_signal = signal_with_gaps[valid_mask]
    
    # Compute triads from valid portions to understand process characteristics
    if bit_mapping == "D1":
        valid_bits = bit_mapping_D1(valid_signal)
    elif bit_mapping == "D2":
        valid_bits = bit_mapping_D2(valid_signal)
    else:
        raise ValueError(f"Unknown bit mapping: {bit_mapping}")
    
    valid_triads = triad_series(valid_bits, beta=BETA)
    
    # Extract characteristic statistics from triads
    mean_H = np.mean(valid_triads[:, 0])
    mean_C = np.mean(valid_triads[:, 1]) 
    mean_S = np.mean(valid_triads[:, 2])
    std_H = np.std(valid_triads[:, 0])
    
    # Use triad characteristics to guide reconstruction
    # Start with cubic interpolation as base
    base_reconstruction = cubic_interpolation(signal_with_gaps, missing_indices)
    
    # Apply triad-informed adjustments
    gap_signal = base_reconstruction[missing_indices]
    
    # Compute triads for the gap region (use larger context)
    context_start = max(0, missing_indices[0] - 200)
    context_end = min(len(base_reconstruction), missing_indices[-1] + 200)
    context_signal = base_reconstruction[context_start:context_end]
    
    if bit_mapping == "D1":
        context_bits = bit_mapping_D1(context_signal)
    else:
        context_bits = bit_mapping_D2(context_signal)
    
    context_triads = triad_series(context_bits, beta=BETA)
    
    # Find where gap region falls in context triads
    gap_start_in_context = missing_indices[0] - context_start
    gap_end_in_context = missing_indices[-1] - context_start
    
    # Adjust gap values to match expected triad characteristics
    if gap_start_in_context < len(context_triads) and gap_end_in_context >= 0:
        # Scale gap values to match expected entropy characteristics
        gap_entropy_target = mean_H
        if len(gap_signal) > 0:
            gap_mean = np.mean(gap_signal)
            gap_std = np.std(gap_signal) if np.std(gap_signal) > 0 else std_H
            
            # Apply scaling based on target entropy and stability
            stability_factor = mean_S
            scaling = (1 + stability_factor * (gap_entropy_target / (gap_std + 1e-8) - 1))
            gap_signal_adjusted = gap_mean + scaling * (gap_signal - gap_mean)
            
            signal_reconstructed[missing_indices] = gap_signal_adjusted
    
    return signal_reconstructed

def continuity_constrained_reconstruction(signal_with_gaps: np.ndarray, missing_indices: np.ndarray,
                                        smoothing_factor: float = 0.1) -> np.ndarray:
    """Reconstruction with explicit continuity constraints."""
    # Start with cubic interpolation
    base_reconstruction = cubic_interpolation(signal_with_gaps, missing_indices)
    
    # Apply smoothing to enforce continuity
    signal_reconstructed = base_reconstruction.copy()
    
    # Apply Gaussian smoothing to the gap region and its neighborhood
    neighborhood_size = len(missing_indices) // 2 + 50
    start_smooth = max(0, missing_indices[0] - neighborhood_size)
    end_smooth = min(len(signal_reconstructed), missing_indices[-1] + neighborhood_size)
    
    # Extract neighborhood
    neighborhood = signal_reconstructed[start_smooth:end_smooth].copy()
    
    # Apply Gaussian smoothing
    sigma = smoothing_factor * len(missing_indices)
    smoothed_neighborhood = gaussian_filter1d(neighborhood, sigma=sigma)
    
    # Replace only the gap region with smoothed values
    gap_start_in_neighborhood = missing_indices[0] - start_smooth
    gap_end_in_neighborhood = missing_indices[-1] - start_smooth + 1
    
    signal_reconstructed[missing_indices] = smoothed_neighborhood[gap_start_in_neighborhood:gap_end_in_neighborhood]
    
    return signal_reconstructed

def evaluate_reconstruction_quality(original: np.ndarray, reconstructed: np.ndarray, 
                                   missing_indices: np.ndarray) -> dict:
    """Evaluate reconstruction quality metrics."""
    # RMSE on missing region
    gap_rmse = rmse(original[missing_indices], reconstructed[missing_indices])
    
    # RMSE on full signal (for context)
    full_rmse = rmse(original, reconstructed)
    
    # Continuity metrics: measure smoothness at gap boundaries
    def boundary_smoothness(signal, gap_indices):
        if len(gap_indices) == 0:
            return 0.0
        
        start_idx = gap_indices[0]
        end_idx = gap_indices[-1]
        
        smoothness = 0.0
        count = 0
        
        # Check continuity at gap start
        if start_idx > 0:
            diff_start = abs(signal[start_idx] - signal[start_idx - 1])
            smoothness += diff_start
            count += 1
        
        # Check continuity at gap end  
        if end_idx < len(signal) - 1:
            diff_end = abs(signal[end_idx + 1] - signal[end_idx])
            smoothness += diff_end
            count += 1
        
        return smoothness / count if count > 0 else 0.0
    
    boundary_roughness = boundary_smoothness(reconstructed, missing_indices)
    
    # Internal gap smoothness (variance of second differences)
    if len(missing_indices) > 2:
        gap_values = reconstructed[missing_indices]
        second_diffs = np.diff(gap_values, n=2)
        internal_roughness = np.var(second_diffs)
    else:
        internal_roughness = 0.0
    
    return {
        'gap_rmse': gap_rmse,
        'full_rmse': full_rmse,
        'boundary_roughness': boundary_roughness,
        'internal_roughness': internal_roughness
    }

def run_reconstruction_test(process_type: str, gap_size: int, gap_position: str, 
                          bit_mapping: str, seed: int = RANDOM_SEED) -> dict:
    """Run reconstruction test for given parameters."""
    print(f"Running T4 test: {process_type}, gap_size={gap_size}, pos={gap_position}, {bit_mapping}")
    
    np.random.seed(seed)
    
    # Generate process
    if process_type == "poisson":
        signal = generate_poisson_process(PROCESS_LENGTH, rate=1.0, seed=seed)
    elif process_type == "van_der_pol":
        signal = generate_van_der_pol(PROCESS_LENGTH, mu=1.0, seed=seed)
    elif process_type == "chirp":
        signal = generate_chirp(PROCESS_LENGTH, f0=0.01, f1=0.05, seed=seed)
    else:
        raise ValueError(f"Unknown process type: {process_type}")
    
    # Create missing data gap
    signal_with_gaps, missing_mask, missing_indices = create_missing_data_gap(
        signal, gap_size, gap_position)
    
    results = {
        'process_type': process_type,
        'gap_size': gap_size,
        'gap_position': gap_position,
        'bit_mapping': bit_mapping,
        'seed': seed,
        'missing_indices': missing_indices.tolist(),
        'reconstructions': {}
    }
    
    # Test each reconstruction method
    for method in RECONSTRUCTION_METHODS:
        print(f"  Testing {method} reconstruction")
        
        if method == "linear":
            reconstructed = linear_interpolation(signal_with_gaps, missing_indices)
        elif method == "cubic":
            reconstructed = cubic_interpolation(signal_with_gaps, missing_indices)
        elif method == "triad_informed":
            reconstructed = triad_informed_reconstruction(signal_with_gaps, missing_indices, bit_mapping)
        elif method == "constrained":
            reconstructed = continuity_constrained_reconstruction(signal_with_gaps, missing_indices)
        
        # Evaluate reconstruction
        quality_metrics = evaluate_reconstruction_quality(signal, reconstructed, missing_indices)
        
        results['reconstructions'][method] = {
            'reconstructed_signal': reconstructed.tolist(),
            'quality_metrics': quality_metrics
        }
        
        print(f"    Gap RMSE: {quality_metrics['gap_rmse']:.6f}")
        print(f"    Boundary roughness: {quality_metrics['boundary_roughness']:.6f}")
    
    return results

def evaluate_hypotheses(results: list) -> dict:
    """Evaluate H7 and H8 hypotheses."""
    
    # H7: Triad-informed reconstruction outperforms naive interpolation
    linear_rmses = []
    triad_rmses = []
    improvement_ratios_h7 = []
    
    # H8: Continuity constraints improve reconstruction  
    cubic_rmses = []
    constrained_rmses = []
    improvement_ratios_h8 = []
    
    for res in results:
        reconstructions = res['reconstructions']
        
        # H7 comparison: Linear vs Triad-informed
        if 'linear' in reconstructions and 'triad_informed' in reconstructions:
            linear_rmse = reconstructions['linear']['quality_metrics']['gap_rmse']
            triad_rmse = reconstructions['triad_informed']['quality_metrics']['gap_rmse']
            
            linear_rmses.append(linear_rmse)
            triad_rmses.append(triad_rmse)
            
            if linear_rmse > 0:
                improvement = (linear_rmse - triad_rmse) / linear_rmse
                improvement_ratios_h7.append(improvement)
        
        # H8 comparison: Cubic vs Constrained
        if 'cubic' in reconstructions and 'constrained' in reconstructions:
            cubic_rmse = reconstructions['cubic']['quality_metrics']['gap_rmse']
            constrained_rmse = reconstructions['constrained']['quality_metrics']['gap_rmse']
            
            cubic_rmses.append(cubic_rmse)
            constrained_rmses.append(constrained_rmse)
            
            if cubic_rmse > 0:
                improvement = (cubic_rmse - constrained_rmse) / cubic_rmse
                improvement_ratios_h8.append(improvement)
    
    # Evaluate H7
    median_improvement_h7 = np.median(improvement_ratios_h7) if improvement_ratios_h7 else 0
    h7_pass = median_improvement_h7 >= IMPROVEMENT_THRESHOLD_H7
    
    # Evaluate H8
    median_improvement_h8 = np.median(improvement_ratios_h8) if improvement_ratios_h8 else 0
    h8_pass = median_improvement_h8 >= IMPROVEMENT_THRESHOLD_H8
    
    evaluation = {
        'H7_triad_informed_outperforms': {
            'median_linear_rmse': np.median(linear_rmses) if linear_rmses else float('inf'),
            'median_triad_rmse': np.median(triad_rmses) if triad_rmses else float('inf'),
            'median_improvement_ratio': median_improvement_h7,
            'threshold': IMPROVEMENT_THRESHOLD_H7,
            'n_comparisons': len(improvement_ratios_h7),
            'pass': h7_pass
        },
        'H8_continuity_improves': {
            'median_cubic_rmse': np.median(cubic_rmses) if cubic_rmses else float('inf'),
            'median_constrained_rmse': np.median(constrained_rmses) if constrained_rmses else float('inf'),
            'median_improvement_ratio': median_improvement_h8,
            'threshold': IMPROVEMENT_THRESHOLD_H8,
            'n_comparisons': len(improvement_ratios_h8),
            'pass': h8_pass
        },
        'overall_pass': h7_pass and h8_pass
    }
    
    return evaluation

def save_results(results: list, evaluation: dict):
    """Save results to files."""
    create_results_dir()
    
    # Save detailed results to JSON (excluding large signal arrays)
    output_data = {
        'test': 'T4_retrodictive_reconstruction',
        'parameters': {
            'process_length': PROCESS_LENGTH,
            'beta': BETA,
            'gap_sizes': GAP_SIZES,
            'gap_positions': GAP_POSITIONS,
            'reconstruction_methods': RECONSTRUCTION_METHODS,
            'improvement_threshold_h7': IMPROVEMENT_THRESHOLD_H7,
            'improvement_threshold_h8': IMPROVEMENT_THRESHOLD_H8
        },
        'results_summary': [],  # Will contain results without signal arrays
        'evaluation': evaluation
    }
    
    # Create summarized results without large arrays
    for res in results:
        summary_res = res.copy()
        # Remove large signal arrays, keep only metrics
        summary_reconstructions = {}
        for method, method_data in res['reconstructions'].items():
            summary_reconstructions[method] = {
                'quality_metrics': method_data['quality_metrics']
            }
        summary_res['reconstructions'] = summary_reconstructions
        output_data['results_summary'].append(summary_res)
    
    with open('results/T4_summary.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save metrics to CSV
    with open('results/T4_reconstruction_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['process_type', 'gap_size', 'gap_position', 'bit_mapping', 
                        'reconstruction_method', 'gap_rmse', 'full_rmse',
                        'boundary_roughness', 'internal_roughness'])
        
        for res in results:
            for method, method_data in res['reconstructions'].items():
                metrics = method_data['quality_metrics']
                writer.writerow([
                    res['process_type'], res['gap_size'], res['gap_position'],
                    res['bit_mapping'], method,
                    metrics['gap_rmse'], metrics['full_rmse'],
                    metrics['boundary_roughness'], metrics['internal_roughness']
                ])

def create_plots(results: list, evaluation: dict):
    """Create visualization plots."""
    create_results_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('T4: Retrodictive Reconstruction Results')
    
    # Collect data for plotting
    method_rmses = {method: [] for method in RECONSTRUCTION_METHODS}
    h7_improvements = []
    h8_improvements = []
    
    for res in results:
        reconstructions = res['reconstructions']
        
        # Collect RMSEs by method
        for method in RECONSTRUCTION_METHODS:
            if method in reconstructions:
                method_rmses[method].append(reconstructions[method]['quality_metrics']['gap_rmse'])
        
        # H7 improvements (Linear vs Triad-informed)
        if 'linear' in reconstructions and 'triad_informed' in reconstructions:
            linear_rmse = reconstructions['linear']['quality_metrics']['gap_rmse']
            triad_rmse = reconstructions['triad_informed']['quality_metrics']['gap_rmse']
            if linear_rmse > 0:
                improvement = (linear_rmse - triad_rmse) / linear_rmse
                h7_improvements.append(improvement)
        
        # H8 improvements (Cubic vs Constrained)
        if 'cubic' in reconstructions and 'constrained' in reconstructions:
            cubic_rmse = reconstructions['cubic']['quality_metrics']['gap_rmse']
            constrained_rmse = reconstructions['constrained']['quality_metrics']['gap_rmse']
            if cubic_rmse > 0:
                improvement = (cubic_rmse - constrained_rmse) / cubic_rmse
                h8_improvements.append(improvement)
    
    # Plot 1: Method comparison (box plot)
    ax = axes[0, 0]
    box_data = [method_rmses[method] for method in RECONSTRUCTION_METHODS if method_rmses[method]]
    box_labels = [method for method in RECONSTRUCTION_METHODS if method_rmses[method]]
    
    if box_data:
        ax.boxplot(box_data, labels=box_labels)
    ax.set_ylabel('Gap RMSE')
    ax.set_title('Reconstruction Method Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: H7 improvement distribution
    ax = axes[0, 1]
    if h7_improvements:
        ax.hist(h7_improvements, bins=20, alpha=0.7, density=True)
        ax.axvline(x=IMPROVEMENT_THRESHOLD_H7, color='r', linestyle='--', 
                   label=f'H7 Threshold ({IMPROVEMENT_THRESHOLD_H7})')
        ax.axvline(x=np.median(h7_improvements), color='g', linestyle='-', 
                   label=f'Median ({np.median(h7_improvements):.3f})')
    ax.set_xlabel('Improvement Ratio (Linear → Triad-informed)')
    ax.set_ylabel('Density')
    ax.set_title('H7: Triad-informed Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: H8 improvement distribution  
    ax = axes[1, 0]
    if h8_improvements:
        ax.hist(h8_improvements, bins=20, alpha=0.7, density=True)
        ax.axvline(x=IMPROVEMENT_THRESHOLD_H8, color='r', linestyle='--',
                   label=f'H8 Threshold ({IMPROVEMENT_THRESHOLD_H8})')
        ax.axvline(x=np.median(h8_improvements), color='g', linestyle='-',
                   label=f'Median ({np.median(h8_improvements):.3f})')
    ax.set_xlabel('Improvement Ratio (Cubic → Constrained)')
    ax.set_ylabel('Density') 
    ax.set_title('H8: Continuity Constraint Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Evaluation summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
Evaluation Summary:

H7 (Triad-informed Outperforms):
  Status: {'PASS' if evaluation['H7_triad_informed_outperforms']['pass'] else 'FAIL'}
  Median Linear RMSE: {evaluation['H7_triad_informed_outperforms']['median_linear_rmse']:.4f}
  Median Triad RMSE: {evaluation['H7_triad_informed_outperforms']['median_triad_rmse']:.4f}
  Median Improvement: {evaluation['H7_triad_informed_outperforms']['median_improvement_ratio']:.4f}
  Threshold: {evaluation['H7_triad_informed_outperforms']['threshold']:.4f}
  Comparisons: {evaluation['H7_triad_informed_outperforms']['n_comparisons']}

H8 (Continuity Improves):
  Status: {'PASS' if evaluation['H8_continuity_improves']['pass'] else 'FAIL'}
  Median Cubic RMSE: {evaluation['H8_continuity_improves']['median_cubic_rmse']:.4f}
  Median Constrained RMSE: {evaluation['H8_continuity_improves']['median_constrained_rmse']:.4f}
  Median Improvement: {evaluation['H8_continuity_improves']['median_improvement_ratio']:.4f}
  Threshold: {evaluation['H8_continuity_improves']['threshold']:.4f}
  Comparisons: {evaluation['H8_continuity_improves']['n_comparisons']}

Overall Test: {'PASS' if evaluation['overall_pass'] else 'FAIL'}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/T4_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the complete T4 test suite."""
    print("="*60)
    print("Running Test T4: Retrodictive Reconstruction With Continuity Constraint")
    print("="*60)
    
    results = []
    
    # Test different process types and configurations
    test_configs = [
        ("poisson", "D1"),
        ("van_der_pol", "D1"),
        ("chirp", "D2"),
        ("van_der_pol", "D2")
    ]
    
    for process_type, bit_mapping in test_configs:
        for gap_size in GAP_SIZES[:2]:  # Limit to first 2 gap sizes for efficiency
            for gap_position in GAP_POSITIONS[:2]:  # Limit positions
                result = run_reconstruction_test(process_type, gap_size, gap_position, 
                                               bit_mapping, RANDOM_SEED)
                results.append(result)
    
    # Evaluate hypotheses
    evaluation = evaluate_hypotheses(results)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"H7 (Triad-informed Outperforms): {'PASS' if evaluation['H7_triad_informed_outperforms']['pass'] else 'FAIL'}")
    print(f"  Median Linear RMSE: {evaluation['H7_triad_informed_outperforms']['median_linear_rmse']:.6f}")
    print(f"  Median Triad RMSE: {evaluation['H7_triad_informed_outperforms']['median_triad_rmse']:.6f}")
    print(f"  Median Improvement Ratio: {evaluation['H7_triad_informed_outperforms']['median_improvement_ratio']:.6f}")
    print(f"  Threshold: {evaluation['H7_triad_informed_outperforms']['threshold']:.6f}")
    print(f"  Comparisons: {evaluation['H7_triad_informed_outperforms']['n_comparisons']}")
    
    print(f"\nH8 (Continuity Improves): {'PASS' if evaluation['H8_continuity_improves']['pass'] else 'FAIL'}")
    print(f"  Median Cubic RMSE: {evaluation['H8_continuity_improves']['median_cubic_rmse']:.6f}")
    print(f"  Median Constrained RMSE: {evaluation['H8_continuity_improves']['median_constrained_rmse']:.6f}")
    print(f"  Median Improvement Ratio: {evaluation['H8_continuity_improves']['median_improvement_ratio']:.6f}")
    print(f"  Threshold: {evaluation['H8_continuity_improves']['threshold']:.6f}")
    print(f"  Comparisons: {evaluation['H8_continuity_improves']['n_comparisons']}")
    
    print(f"\nOVERALL TEST: {'PASS' if evaluation['overall_pass'] else 'FAIL'}")
    
    # Save results
    save_results(results, evaluation)
    create_plots(results, evaluation)
    
    print(f"\nResults saved to results/T4_summary.json and results/T4_reconstruction_metrics.csv")
    print(f"Plots saved to results/T4_plots.png")
    
    return evaluation['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)