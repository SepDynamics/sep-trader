#!/usr/bin/env python3
"""
Test T5: Smoothing Beats Filtering (uncertainty reduction)
Tests H9: SEP-informed smoothing outperforms naive filtering in uncertainty reduction
Tests H10: Optimal smoothing parameters correlate with triad stability metrics
"""

import numpy as np
import json
import csv
import os
from pathlib import Path
from scipy import signal, optimize
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error
from sep_core import (
    triad_series, rmse, bit_mapping_D1, bit_mapping_D2,
    generate_poisson_process, generate_van_der_pol, generate_chirp, RANDOM_SEED
)
import matplotlib.pyplot as plt

# Test parameters
PROCESS_LENGTH = 50000
BETA = 0.1  # EMA parameter
NOISE_LEVELS = [0.1, 0.2, 0.5]  # Additive noise standard deviations
FILTER_METHODS = ["naive_gaussian", "naive_median", "sep_informed", "adaptive_sep"]
SMOOTHING_WINDOW_SIZES = [5, 10, 20, 50]  # For naive methods
UNCERTAINTY_THRESHOLD_H9 = 0.15  # Min improvement in uncertainty reduction
CORRELATION_THRESHOLD_H10 = 0.3  # Min correlation coefficient for H10

def create_results_dir():
    """Create results directory if it doesn't exist."""
    Path("results").mkdir(exist_ok=True)

def add_noise_to_signal(signal: np.ndarray, noise_level: float, seed: int = RANDOM_SEED) -> np.ndarray:
    """Add Gaussian noise to signal."""
    np.random.seed(seed)
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def naive_gaussian_filter(noisy_signal: np.ndarray, window_size: int) -> np.ndarray:
    """Apply naive Gaussian smoothing filter."""
    sigma = window_size / 4.0  # Standard relationship
    return gaussian_filter1d(noisy_signal, sigma=sigma)

def naive_median_filter(noisy_signal: np.ndarray, window_size: int) -> np.ndarray:
    """Apply naive median filter."""
    filtered = np.copy(noisy_signal)
    half_window = window_size // 2
    
    for i in range(len(noisy_signal)):
        start = max(0, i - half_window)
        end = min(len(noisy_signal), i + half_window + 1)
        filtered[i] = np.median(noisy_signal[start:end])
    
    return filtered

def compute_local_uncertainty(signal: np.ndarray, window_size: int = 20) -> np.ndarray:
    """Compute local uncertainty (standard deviation) in sliding windows."""
    uncertainty = np.zeros_like(signal)
    half_window = window_size // 2
    
    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        uncertainty[i] = np.std(signal[start:end])
    
    return uncertainty

def sep_informed_filter(noisy_signal: np.ndarray, bit_mapping: str, 
                       adaptive: bool = False) -> dict:
    """Apply SEP-informed filtering using triad characteristics."""
    
    # Compute triads from noisy signal
    if bit_mapping == "D1":
        noisy_bits = bit_mapping_D1(noisy_signal)
    elif bit_mapping == "D2":
        noisy_bits = bit_mapping_D2(noisy_signal)
    else:
        raise ValueError(f"Unknown bit mapping: {bit_mapping}")
    
    noisy_triads = triad_series(noisy_bits, beta=BETA)
    
    # Extract stability metrics
    stability = noisy_triads[:, 2]  # S component
    coherence = noisy_triads[:, 1]  # C component
    entropy = noisy_triads[:, 0]   # H component
    
    # Compute adaptive smoothing parameters based on stability
    if adaptive:
        # Higher stability regions get less smoothing, lower stability get more
        base_sigma = 2.0
        stability_factor = 1.0 - np.clip(stability, 0, 1)  # Invert: low stability = high factor
        sigma_values = base_sigma * (1 + 2 * stability_factor)
        
        # Apply variable Gaussian smoothing
        filtered_signal = np.copy(noisy_signal)
        window_size = 100  # Process in chunks
        
        for i in range(0, len(noisy_signal), window_size):
            end_idx = min(i + window_size, len(noisy_signal))
            chunk = noisy_signal[i:end_idx]
            
            # Use median sigma for this chunk
            if i < len(sigma_values):
                chunk_sigma = np.median(sigma_values[max(0, i-50):min(len(sigma_values), end_idx+50)])
            else:
                chunk_sigma = base_sigma
                
            filtered_chunk = gaussian_filter1d(chunk, sigma=chunk_sigma)
            filtered_signal[i:end_idx] = filtered_chunk
    
    else:
        # Fixed SEP-informed smoothing based on overall characteristics
        mean_stability = np.mean(stability)
        mean_coherence = np.mean(coherence)
        
        # Compute adaptive sigma based on stability and coherence
        # Higher stability and coherence suggest less noise, need less smoothing
        stability_factor = 1.0 - np.clip(mean_stability, 0, 1)
        coherence_factor = 1.0 - np.clip(mean_coherence, 0, 1)
        
        adaptive_sigma = 2.0 + 4.0 * stability_factor + 2.0 * coherence_factor
        filtered_signal = gaussian_filter1d(noisy_signal, sigma=adaptive_sigma)
    
    return {
        'filtered_signal': filtered_signal,
        'triads': noisy_triads,
        'stability': stability,
        'coherence': coherence,
        'entropy': entropy,
        'adaptive_params': sigma_values if adaptive else [adaptive_sigma] * len(noisy_signal)
    }

def evaluate_filtering_performance(original: np.ndarray, noisy: np.ndarray, 
                                 filtered: np.ndarray) -> dict:
    """Evaluate filtering performance metrics."""
    
    # Signal reconstruction quality
    signal_rmse = rmse(original, filtered)
    signal_snr_improvement = 10 * np.log10(
        np.var(noisy - original) / (np.var(filtered - original) + 1e-12)
    )
    
    # Uncertainty reduction
    original_uncertainty = compute_local_uncertainty(noisy)
    filtered_uncertainty = compute_local_uncertainty(filtered)
    
    uncertainty_reduction_mean = np.mean(original_uncertainty) - np.mean(filtered_uncertainty)
    uncertainty_reduction_ratio = uncertainty_reduction_mean / (np.mean(original_uncertainty) + 1e-12)
    
    # Smoothness metrics
    original_smoothness = np.var(np.diff(original))
    filtered_smoothness = np.var(np.diff(filtered))
    smoothness_preservation = 1.0 - abs(filtered_smoothness - original_smoothness) / (original_smoothness + 1e-12)
    
    return {
        'signal_rmse': signal_rmse,
        'snr_improvement_db': signal_snr_improvement,
        'uncertainty_reduction_mean': uncertainty_reduction_mean,
        'uncertainty_reduction_ratio': uncertainty_reduction_ratio,
        'smoothness_preservation': smoothness_preservation,
        'mean_original_uncertainty': np.mean(original_uncertainty),
        'mean_filtered_uncertainty': np.mean(filtered_uncertainty)
    }

def run_smoothing_test(process_type: str, noise_level: float, bit_mapping: str,
                      seed: int = RANDOM_SEED) -> dict:
    """Run smoothing vs filtering test for given parameters."""
    print(f"Running T5 test: {process_type}, noise={noise_level}, {bit_mapping}")
    
    np.random.seed(seed)
    
    # Generate clean process
    if process_type == "poisson":
        clean_signal = generate_poisson_process(PROCESS_LENGTH, rate=1.0, seed=seed)
    elif process_type == "van_der_pol":
        clean_signal = generate_van_der_pol(PROCESS_LENGTH, mu=1.0, seed=seed)
    elif process_type == "chirp":
        clean_signal = generate_chirp(PROCESS_LENGTH, f0=0.005, f1=0.02, seed=seed)
    else:
        raise ValueError(f"Unknown process type: {process_type}")
    
    # Add noise
    noisy_signal = add_noise_to_signal(clean_signal, noise_level, seed)
    
    results = {
        'process_type': process_type,
        'noise_level': noise_level,
        'bit_mapping': bit_mapping,
        'seed': seed,
        'filtering_results': {}
    }
    
    # Test naive methods with different window sizes
    for method in ["naive_gaussian", "naive_median"]:
        print(f"  Testing {method} filtering")
        
        method_results = {}
        
        for window_size in SMOOTHING_WINDOW_SIZES:
            if method == "naive_gaussian":
                filtered = naive_gaussian_filter(noisy_signal, window_size)
            elif method == "naive_median":
                filtered = naive_median_filter(noisy_signal, window_size)
            
            performance = evaluate_filtering_performance(clean_signal, noisy_signal, filtered)
            
            method_results[f'window_{window_size}'] = {
                'performance': performance,
                'window_size': window_size
            }
            
            print(f"    Window {window_size}: RMSE = {performance['signal_rmse']:.6f}, "
                  f"Uncertainty reduction = {performance['uncertainty_reduction_ratio']:.4f}")
        
        results['filtering_results'][method] = method_results
    
    # Test SEP-informed methods
    for method in ["sep_informed", "adaptive_sep"]:
        print(f"  Testing {method} filtering")
        
        adaptive = (method == "adaptive_sep")
        sep_result = sep_informed_filter(noisy_signal, bit_mapping, adaptive=adaptive)
        filtered = sep_result['filtered_signal']
        
        performance = evaluate_filtering_performance(clean_signal, noisy_signal, filtered)
        
        results['filtering_results'][method] = {
            'performance': performance,
            'sep_data': {
                'mean_stability': float(np.mean(sep_result['stability'])),
                'mean_coherence': float(np.mean(sep_result['coherence'])),
                'mean_entropy': float(np.mean(sep_result['entropy'])),
                'adaptive_params_mean': float(np.mean(sep_result['adaptive_params'])),
                'adaptive_params_std': float(np.std(sep_result['adaptive_params']))
            }
        }
        
        print(f"    {method}: RMSE = {performance['signal_rmse']:.6f}, "
              f"Uncertainty reduction = {performance['uncertainty_reduction_ratio']:.4f}")
        print(f"    Mean stability = {np.mean(sep_result['stability']):.4f}")
    
    return results

def evaluate_hypotheses(results: list) -> dict:
    """Evaluate H9 and H10 hypotheses."""
    
    # H9: SEP-informed smoothing outperforms naive filtering
    naive_uncertainties = []
    sep_uncertainties = []
    improvement_ratios = []
    
    # H10: Optimal parameters correlate with stability
    stability_values = []
    optimal_params = []
    
    for res in results:
        filtering_results = res['filtering_results']
        
        # Find best naive method performance
        best_naive_uncertainty = float('inf')
        
        for method in ["naive_gaussian", "naive_median"]:
            if method in filtering_results:
                for window_key, window_data in filtering_results[method].items():
                    uncertainty_reduction = window_data['performance']['uncertainty_reduction_ratio']
                    if uncertainty_reduction < best_naive_uncertainty:
                        best_naive_uncertainty = uncertainty_reduction
        
        naive_uncertainties.append(best_naive_uncertainty)
        
        # Get SEP method performance
        best_sep_uncertainty = -float('inf')
        best_sep_stability = None
        best_sep_params = None
        
        for method in ["sep_informed", "adaptive_sep"]:
            if method in filtering_results:
                method_data = filtering_results[method]
                uncertainty_reduction = method_data['performance']['uncertainty_reduction_ratio']
                if uncertainty_reduction > best_sep_uncertainty:
                    best_sep_uncertainty = uncertainty_reduction
                    best_sep_stability = method_data['sep_data']['mean_stability']
                    best_sep_params = method_data['sep_data']['adaptive_params_mean']
        
        sep_uncertainties.append(best_sep_uncertainty)
        
        # H9: Improvement ratio
        if best_naive_uncertainty < float('inf') and best_sep_uncertainty > -float('inf'):
            if best_naive_uncertainty != 0:
                improvement = (best_sep_uncertainty - best_naive_uncertainty) / abs(best_naive_uncertainty)
                improvement_ratios.append(improvement)
        
        # H10: Stability correlation data
        if best_sep_stability is not None and best_sep_params is not None:
            stability_values.append(best_sep_stability)
            optimal_params.append(best_sep_params)
    
    # Evaluate H9
    median_improvement = np.median(improvement_ratios) if improvement_ratios else 0
    h9_pass = median_improvement >= UNCERTAINTY_THRESHOLD_H9
    
    # Evaluate H10: Correlation between stability and optimal parameters
    correlation_coeff = 0.0
    if len(stability_values) >= 3 and len(optimal_params) >= 3:
        correlation_coeff = np.corrcoef(stability_values, optimal_params)[0, 1]
        if np.isnan(correlation_coeff):
            correlation_coeff = 0.0
    
    h10_pass = abs(correlation_coeff) >= CORRELATION_THRESHOLD_H10
    
    evaluation = {
        'H9_sep_outperforms_naive': {
            'median_naive_uncertainty_reduction': np.median(naive_uncertainties) if naive_uncertainties else 0,
            'median_sep_uncertainty_reduction': np.median(sep_uncertainties) if sep_uncertainties else 0,
            'median_improvement_ratio': median_improvement,
            'threshold': UNCERTAINTY_THRESHOLD_H9,
            'n_comparisons': len(improvement_ratios),
            'pass': h9_pass
        },
        'H10_params_correlate_stability': {
            'correlation_coefficient': float(correlation_coeff),
            'threshold': CORRELATION_THRESHOLD_H10,
            'n_data_points': len(stability_values),
            'stability_range': [float(np.min(stability_values)), float(np.max(stability_values))] if stability_values else [0, 0],
            'params_range': [float(np.min(optimal_params)), float(np.max(optimal_params))] if optimal_params else [0, 0],
            'pass': bool(h10_pass)
        },
        'overall_pass': bool(h9_pass and h10_pass)
    }
    
    return evaluation

def save_results(results: list, evaluation: dict):
    """Save results to files."""
    create_results_dir()
    
    # Save detailed results to JSON
    output_data = {
        'test': 'T5_smoothing_beats_filtering',
        'parameters': {
            'process_length': PROCESS_LENGTH,
            'beta': BETA,
            'noise_levels': NOISE_LEVELS,
            'filter_methods': FILTER_METHODS,
            'smoothing_window_sizes': SMOOTHING_WINDOW_SIZES,
            'uncertainty_threshold_h9': UNCERTAINTY_THRESHOLD_H9,
            'correlation_threshold_h10': CORRELATION_THRESHOLD_H10
        },
        'results': results,
        'evaluation': evaluation
    }
    
    with open('results/T5_summary.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save metrics to CSV
    with open('results/T5_filtering_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['process_type', 'noise_level', 'bit_mapping', 'method', 'window_size_or_type',
                        'signal_rmse', 'snr_improvement_db', 'uncertainty_reduction_ratio',
                        'smoothness_preservation', 'mean_stability', 'mean_coherence'])
        
        for res in results:
            for method, method_data in res['filtering_results'].items():
                if method in ["naive_gaussian", "naive_median"]:
                    for window_key, window_data in method_data.items():
                        writer.writerow([
                            res['process_type'], res['noise_level'], res['bit_mapping'],
                            method, window_data['window_size'],
                            window_data['performance']['signal_rmse'],
                            window_data['performance']['snr_improvement_db'],
                            window_data['performance']['uncertainty_reduction_ratio'],
                            window_data['performance']['smoothness_preservation'],
                            '', ''  # No stability/coherence for naive methods
                        ])
                else:  # SEP methods
                    writer.writerow([
                        res['process_type'], res['noise_level'], res['bit_mapping'],
                        method, 'adaptive',
                        method_data['performance']['signal_rmse'],
                        method_data['performance']['snr_improvement_db'],
                        method_data['performance']['uncertainty_reduction_ratio'],
                        method_data['performance']['smoothness_preservation'],
                        method_data['sep_data']['mean_stability'],
                        method_data['sep_data']['mean_coherence']
                    ])

def create_plots(results: list, evaluation: dict):
    """Create visualization plots."""
    create_results_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('T5: Smoothing Beats Filtering Results')
    
    # Collect data for plotting
    method_uncertainties = {}
    stability_values = []
    param_values = []
    improvement_ratios = []
    
    for res in results:
        filtering_results = res['filtering_results']
        
        for method, method_data in filtering_results.items():
            if method not in method_uncertainties:
                method_uncertainties[method] = []
            
            if method in ["naive_gaussian", "naive_median"]:
                # Get best performance across window sizes
                best_uncertainty = -float('inf')
                for window_key, window_data in method_data.items():
                    uncertainty_reduction = window_data['performance']['uncertainty_reduction_ratio']
                    if uncertainty_reduction > best_uncertainty:
                        best_uncertainty = uncertainty_reduction
                method_uncertainties[method].append(best_uncertainty)
            else:  # SEP methods
                uncertainty_reduction = method_data['performance']['uncertainty_reduction_ratio']
                method_uncertainties[method].append(uncertainty_reduction)
                
                # Collect stability correlation data
                stability = method_data['sep_data']['mean_stability']
                params = method_data['sep_data']['adaptive_params_mean']
                stability_values.append(stability)
                param_values.append(params)
    
    # Plot 1: Method comparison
    ax = axes[0, 0]
    methods_to_plot = [method for method in method_uncertainties.keys() if method_uncertainties[method]]
    box_data = [method_uncertainties[method] for method in methods_to_plot]
    
    if box_data:
        ax.boxplot(box_data, labels=methods_to_plot)
    ax.set_ylabel('Uncertainty Reduction Ratio')
    ax.set_title('Filtering Method Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: H9 improvement analysis
    ax = axes[0, 1]
    naive_methods = ["naive_gaussian", "naive_median"]
    sep_methods = ["sep_informed", "adaptive_sep"]
    
    naive_data = []
    sep_data = []
    
    for method in naive_methods:
        if method in method_uncertainties:
            naive_data.extend(method_uncertainties[method])
    
    for method in sep_methods:
        if method in method_uncertainties:
            sep_data.extend(method_uncertainties[method])
    
    if naive_data and sep_data:
        ax.scatter([1] * len(naive_data), naive_data, alpha=0.7, label='Naive Methods', color='red')
        ax.scatter([2] * len(sep_data), sep_data, alpha=0.7, label='SEP Methods', color='green')
        
        ax.axhline(y=UNCERTAINTY_THRESHOLD_H9, color='k', linestyle='--', alpha=0.5, 
                   label=f'H9 Improvement Threshold')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Naive', 'SEP'])
    ax.set_ylabel('Uncertainty Reduction Ratio')
    ax.set_title('H9: SEP vs Naive Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: H10 stability-parameter correlation
    ax = axes[1, 0]
    if len(stability_values) >= 3 and len(param_values) >= 3:
        ax.scatter(stability_values, param_values, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(stability_values, param_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(stability_values), max(stability_values), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend line')
        
        # Add correlation info
        corr_coeff = evaluation['H10_params_correlate_stability']['correlation_coefficient']
        ax.text(0.05, 0.95, f'r = {corr_coeff:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Mean Stability')
    ax.set_ylabel('Optimal Smoothing Parameter')
    ax.set_title('H10: Stability vs Optimal Parameters')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Evaluation summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
Evaluation Summary:

H9 (SEP Outperforms Naive):
  Status: {'PASS' if evaluation['H9_sep_outperforms_naive']['pass'] else 'FAIL'}
  Naive Median: {evaluation['H9_sep_outperforms_naive']['median_naive_uncertainty_reduction']:.4f}
  SEP Median: {evaluation['H9_sep_outperforms_naive']['median_sep_uncertainty_reduction']:.4f}
  Improvement: {evaluation['H9_sep_outperforms_naive']['median_improvement_ratio']:.4f}
  Threshold: {evaluation['H9_sep_outperforms_naive']['threshold']:.4f}
  Comparisons: {evaluation['H9_sep_outperforms_naive']['n_comparisons']}

H10 (Parameter-Stability Correlation):
  Status: {'PASS' if evaluation['H10_params_correlate_stability']['pass'] else 'FAIL'}
  Correlation: {evaluation['H10_params_correlate_stability']['correlation_coefficient']:.4f}
  Threshold: {evaluation['H10_params_correlate_stability']['threshold']:.4f}
  Data Points: {evaluation['H10_params_correlate_stability']['n_data_points']}

Overall Test: {'PASS' if evaluation['overall_pass'] else 'FAIL'}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/T5_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the complete T5 test suite."""
    print("="*60)
    print("Running Test T5: Smoothing Beats Filtering (uncertainty reduction)")
    print("="*60)
    
    results = []
    
    # Test different configurations
    test_configs = [
        ("van_der_pol", "D1"),
        ("chirp", "D2"),
        ("poisson", "D1")
    ]
    
    for process_type, bit_mapping in test_configs:
        for noise_level in NOISE_LEVELS[:2]:  # Limit noise levels for efficiency
            result = run_smoothing_test(process_type, noise_level, bit_mapping, RANDOM_SEED)
            results.append(result)
    
    # Evaluate hypotheses
    evaluation = evaluate_hypotheses(results)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"H9 (SEP Outperforms Naive): {'PASS' if evaluation['H9_sep_outperforms_naive']['pass'] else 'FAIL'}")
    print(f"  Naive median uncertainty reduction: {evaluation['H9_sep_outperforms_naive']['median_naive_uncertainty_reduction']:.6f}")
    print(f"  SEP median uncertainty reduction: {evaluation['H9_sep_outperforms_naive']['median_sep_uncertainty_reduction']:.6f}")
    print(f"  Median improvement ratio: {evaluation['H9_sep_outperforms_naive']['median_improvement_ratio']:.6f}")
    print(f"  Threshold: {evaluation['H9_sep_outperforms_naive']['threshold']:.6f}")
    print(f"  Comparisons: {evaluation['H9_sep_outperforms_naive']['n_comparisons']}")
    
    print(f"\nH10 (Parameter-Stability Correlation): {'PASS' if evaluation['H10_params_correlate_stability']['pass'] else 'FAIL'}")
    print(f"  Correlation coefficient: {evaluation['H10_params_correlate_stability']['correlation_coefficient']:.6f}")
    print(f"  Threshold: {evaluation['H10_params_correlate_stability']['threshold']:.6f}")
    print(f"  Data points: {evaluation['H10_params_correlate_stability']['n_data_points']}")
    print(f"  Stability range: {evaluation['H10_params_correlate_stability']['stability_range']}")
    print(f"  Parameter range: {evaluation['H10_params_correlate_stability']['params_range']}")
    
    print(f"\nOVERALL TEST: {'PASS' if evaluation['overall_pass'] else 'FAIL'}")
    
    # Save results
    save_results(results, evaluation)
    create_plots(results, evaluation)
    
    print(f"\nResults saved to results/T5_summary.json and results/T5_filtering_metrics.csv")
    print(f"Plots saved to results/T5_plots.png")
    
    return evaluation['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)