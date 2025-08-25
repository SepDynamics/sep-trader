#!/usr/bin/env python3
"""
Test T1: Isolated vs Reactive Time Scaling
Tests H1: Isolated processes show invariant triad trajectories under time scaling
Tests H2: Reactive processes break triad alignment under naive time scaling
"""

import numpy as np
import json
import csv
import os
from pathlib import Path
from sep_core import (
    triad_series, rmse, bit_mapping_D1, bit_mapping_D2,
    generate_poisson_process, generate_van_der_pol, RANDOM_SEED
)
import matplotlib.pyplot as plt

# Test parameters
PROCESS_LENGTH = 200000
BETA = 0.1  # EMA parameter (half-life ~64 steps: beta = 1 - exp(-ln(2)/64))
GAMMA_VALUES = [1.2, 1.5, 2.0]  # Time scaling factors
RMSE_THRESHOLD_H1 = 0.05  # Pass threshold for isolated processes
MIN_REACTIVE_RATIO = 2.0  # Reactive RMSE should be >= 2x isolated RMSE
PRIMARY_MAPPING = "D2"  # use D2 for formal H1/H2 evaluation; D1 reported as sensitivity

def create_results_dir():
    """Create results directory if it doesn't exist."""
    Path("results").mkdir(exist_ok=True)

def time_scale_signal(signal: np.ndarray, gamma: float) -> np.ndarray:
    """Apply time scaling by factor gamma: x_gamma(t) = x(t/gamma)."""
    n = len(signal)
    
    # Create new time indices scaled by gamma
    # For gamma > 1: signal compressed (faster dynamics)
    # For gamma < 1: signal stretched (slower dynamics)
    original_time = np.arange(n)
    scaled_time = original_time * gamma
    
    # Interpolate the original signal at the scaled time points
    # Use extrapolation for points beyond the original range
    scaled_signal = np.interp(scaled_time, original_time, signal,
                             left=signal[0], right=signal[-1])
    
    return scaled_signal

def align_triad_curves(triads_orig: np.ndarray, triads_scaled: np.ndarray, gamma: float) -> np.ndarray:
    """Align triad curves by evaluating the scaled curve at orig_time/gamma."""
    n_orig = len(triads_orig)
    x_orig = np.linspace(0.0, 1.0, n_orig)
    x_scaled = np.linspace(0.0, 1.0, len(triads_scaled))
    x_query = np.clip(x_orig / gamma, 0.0, 1.0)

    aligned = np.zeros_like(triads_orig)
    for i in range(3):
        aligned[:, i] = np.interp(x_query, x_scaled, triads_scaled[:, i],
                                  left=triads_scaled[0, i], right=triads_scaled[-1, i])
    return aligned

def compute_joint_rmse(triads1: np.ndarray, triads2: np.ndarray) -> float:
    """Compute joint RMSE as average of component RMSEs for fairer weighting."""
    h = rmse(triads1[:,0], triads2[:,0])
    c = rmse(triads1[:,1], triads2[:,1])
    s = rmse(triads1[:,2], triads2[:,2])
    return float((h + c + s) / 3.0)

def run_time_scaling_test(process_type: str, bit_mapping: str, seed: int = RANDOM_SEED) -> dict:
    """Run time scaling test for given process type and bit mapping."""
    np.random.seed(seed)
    
    print(f"Running T1 test: {process_type} process with {bit_mapping} mapping")
    
    # Generate process
    if process_type == "isolated":
        signal = generate_poisson_process(PROCESS_LENGTH, rate=1.0, seed=seed)
    elif process_type == "reactive":
        signal = generate_van_der_pol(PROCESS_LENGTH, mu=1.0, seed=seed)
    else:
        raise ValueError(f"Unknown process type: {process_type}")
    
    # Apply bit mapping
    if bit_mapping == "D1":
        bits_orig = bit_mapping_D1(signal)
    elif bit_mapping == "D2":
        bits_orig = bit_mapping_D2(signal)
    else:
        raise ValueError(f"Unknown bit mapping: {bit_mapping}")
    
    # Compute original triads
    triads_orig = triad_series(bits_orig, beta=BETA)
    
    results = {
        'process_type': process_type,
        'bit_mapping': bit_mapping,
        'seed': seed,
        'gamma_rmses': {},
        'individual_rmses': {}
    }
    
    # Test each scaling factor
    for gamma in GAMMA_VALUES:
        print(f"  Testing gamma = {gamma}")
        
        # Create time-scaled signal
        signal_scaled = time_scale_signal(signal, gamma)
        
        # Apply same bit mapping to scaled signal
        if bit_mapping == "D1":
            bits_scaled = bit_mapping_D1(signal_scaled)
        elif bit_mapping == "D2":
            bits_scaled = bit_mapping_D2(signal_scaled)
        
        # Compute scaled triads
        triads_scaled = triad_series(bits_scaled, beta=BETA)
        
        # Align curves
        triads_aligned = align_triad_curves(triads_orig, triads_scaled, gamma)
        
        # Compute RMSEs
        joint_rmse = compute_joint_rmse(triads_orig, triads_aligned)
        h_rmse = rmse(triads_orig[:, 0], triads_aligned[:, 0])
        c_rmse = rmse(triads_orig[:, 1], triads_aligned[:, 1])
        s_rmse = rmse(triads_orig[:, 2], triads_aligned[:, 2])
        
        # Convert to native Python types to avoid JSON serialization issues
        results['gamma_rmses'][float(gamma)] = float(joint_rmse)
        results['individual_rmses'][float(gamma)] = {
            'H': float(h_rmse),
            'C': float(c_rmse),
            'S': float(s_rmse)
        }
        
        print(f"    Joint RMSE: {joint_rmse:.6f}")
        print(f"    Individual RMSEs - H: {h_rmse:.6f}, C: {c_rmse:.6f}, S: {s_rmse:.6f}")
    
    return results

def evaluate_hypotheses(results: dict) -> dict:
    """Evaluate H1 and H2 hypotheses from test results."""
    # Group by process and mapping
    groups = {(r['process_type'], r['bit_mapping']): r for r in results}

    def median_joint_rmse(proc, mapping):
        res = groups.get((proc, mapping))
        if not res: return float('inf')
        vals = list(res['gamma_rmses'].values())
        return float(np.median(vals)) if vals else float('inf')

    # H1: isolated invariance on PRIMARY_MAPPING only
    isolated_med = median_joint_rmse('isolated', PRIMARY_MAPPING)
    h1_pass = isolated_med <= RMSE_THRESHOLD_H1

    # H2: reactive breaks relative to same mapping baseline
    reactive_med = median_joint_rmse('reactive', PRIMARY_MAPPING)
    ratio = reactive_med / isolated_med if isolated_med > 0 else float('inf')
    h2_pass = ratio >= MIN_REACTIVE_RATIO

    evaluation = {
        'H1_isolated_invariance': {
            'median_joint_rmse': isolated_med,
            'threshold': RMSE_THRESHOLD_H1,
            'mapping': PRIMARY_MAPPING,
            'pass': bool(h1_pass),
        },
        'H2_reactive_breaks': {
            'median_joint_rmse': reactive_med,
            'isolated_median': isolated_med,
            'ratio': ratio,
            'min_ratio_threshold': MIN_REACTIVE_RATIO,
            'mapping': PRIMARY_MAPPING,
            'pass': bool(h2_pass),
        },
        # also report sensitivity (non-blocking)
        'sensitivity': {
            m: {
                'isolated_median': median_joint_rmse('isolated', m),
                'reactive_median': median_joint_rmse('reactive', m),
                'ratio': (median_joint_rmse('reactive', m) /
                          max(1e-12, median_joint_rmse('isolated', m)))
            } for m in ['D1', 'D2']
        },
        'overall_pass': bool(h1_pass and h2_pass)
    }
    return evaluation

def convert_to_native_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):      # <-- add this
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

def save_results(results: list, evaluation: dict):
    """Save results to CSV and JSON files."""
    create_results_dir()
    
    # Convert NumPy types to native Python types for JSON serialization
    results_clean = convert_to_native_types(results)
    evaluation_clean = convert_to_native_types(evaluation)
    
    # Save detailed results to JSON
    output_data = {
        'test': 'T1_time_scaling',
        'parameters': {
            'process_length': int(PROCESS_LENGTH),
            'beta': float(BETA),
            'gamma_values': [float(g) for g in GAMMA_VALUES],
            'rmse_threshold_h1': float(RMSE_THRESHOLD_H1),
            'min_reactive_ratio': float(MIN_REACTIVE_RATIO)
        },
        'results': results_clean,
        'evaluation': evaluation_clean
    }
    
    with open('results/T1_summary.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save metrics to CSV
    with open('results/T1_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['process_type', 'bit_mapping', 'seed', 'gamma', 
                        'joint_rmse', 'H_rmse', 'C_rmse', 'S_rmse'])
        
        for res in results:
            for gamma in GAMMA_VALUES:
                writer.writerow([
                    res['process_type'],
                    res['bit_mapping'],
                    res['seed'],
                    gamma,
                    res['gamma_rmses'][gamma],
                    res['individual_rmses'][gamma]['H'],
                    res['individual_rmses'][gamma]['C'],
                    res['individual_rmses'][gamma]['S']
                ])

def create_plots(results: list):
    """Create visualization plots."""
    create_results_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('T1: Isolated vs Reactive Time Scaling Results')
    
    # Collect data by process type
    isolated_data = [r for r in results if r['process_type'] == 'isolated']
    reactive_data = [r for r in results if r['process_type'] == 'reactive']
    
    # Plot joint RMSE vs gamma
    ax = axes[0, 0]
    for res in isolated_data:
        gammas = list(res['gamma_rmses'].keys())
        rmses = list(res['gamma_rmses'].values())
        ax.plot(gammas, rmses, 'b-o', alpha=0.7, label='Isolated' if res == isolated_data[0] else "")
    
    for res in reactive_data:
        gammas = list(res['gamma_rmses'].keys())
        rmses = list(res['gamma_rmses'].values())
        ax.plot(gammas, rmses, 'r-s', alpha=0.7, label='Reactive' if res == reactive_data[0] else "")
    
    ax.axhline(y=RMSE_THRESHOLD_H1, color='k', linestyle='--', alpha=0.5, label=f'H1 Threshold ({RMSE_THRESHOLD_H1})')
    ax.set_xlabel('Time Scaling Factor (γ)')
    ax.set_ylabel('Joint RMSE')
    ax.set_title('Joint RMSE vs Time Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax = axes[0, 1]
    isolated_rmses = []
    reactive_rmses = []
    
    for res in isolated_data:
        isolated_rmses.extend(list(res['gamma_rmses'].values()))
    for res in reactive_data:
        reactive_rmses.extend(list(res['gamma_rmses'].values()))
    
    ax.boxplot([isolated_rmses, reactive_rmses], labels=['Isolated', 'Reactive'])
    ax.axhline(y=RMSE_THRESHOLD_H1, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel('Joint RMSE')
    ax.set_title('RMSE Distribution Comparison')
    ax.grid(True, alpha=0.3)
    
    # Individual component RMSEs
    ax = axes[1, 0]
    components = ['H', 'C', 'S']
    isolated_comp_rmses = {comp: [] for comp in components}
    reactive_comp_rmses = {comp: [] for comp in components}
    
    for res in isolated_data:
        for gamma in GAMMA_VALUES:
            for comp in components:
                isolated_comp_rmses[comp].append(res['individual_rmses'][gamma][comp])
    
    for res in reactive_data:
        for gamma in GAMMA_VALUES:
            for comp in components:
                reactive_comp_rmses[comp].append(res['individual_rmses'][gamma][comp])
    
    x_pos = np.arange(len(components))
    width = 0.35
    
    isolated_means = [np.mean(isolated_comp_rmses[comp]) for comp in components]
    reactive_means = [np.mean(reactive_comp_rmses[comp]) for comp in components]
    
    ax.bar(x_pos - width/2, isolated_means, width, label='Isolated', alpha=0.7)
    ax.bar(x_pos + width/2, reactive_means, width, label='Reactive', alpha=0.7)
    
    ax.set_xlabel('Triad Component')
    ax.set_ylabel('Mean RMSE')
    ax.set_title('Component-wise RMSE Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(components)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ratio plot
    ax = axes[1, 1]
    ratios = []
    for gamma in GAMMA_VALUES:
        isolated_gamma = [r['gamma_rmses'][gamma] for r in isolated_data]
        reactive_gamma = [r['gamma_rmses'][gamma] for r in reactive_data]
        
        if isolated_gamma and reactive_gamma:
            ratio = np.mean(reactive_gamma) / np.mean(isolated_gamma)
            ratios.append(ratio)
        else:
            ratios.append(1.0)
    
    ax.plot(GAMMA_VALUES, ratios, 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=MIN_REACTIVE_RATIO, color='r', linestyle='--', alpha=0.7, 
               label=f'H2 Threshold ({MIN_REACTIVE_RATIO})')
    ax.set_xlabel('Time Scaling Factor (γ)')
    ax.set_ylabel('Reactive/Isolated RMSE Ratio')
    ax.set_title('H2: Reactive vs Isolated RMSE Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/T1_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the complete T1 test suite."""
    print("="*60)
    print("Running Test T1: Isolated vs Reactive Time Scaling")
    print("="*60)
    
    results = []
    
    # Test isolated processes
    for bit_mapping in ["D1", "D2"]:
        result = run_time_scaling_test("isolated", bit_mapping, seed=RANDOM_SEED)
        results.append(result)
    
    # Test reactive processes  
    for bit_mapping in ["D1", "D2"]:
        result = run_time_scaling_test("reactive", bit_mapping, seed=RANDOM_SEED + 1)
        results.append(result)
    
    # Evaluate hypotheses
    evaluation = evaluate_hypotheses(results)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"H1 (Isolated Invariance - {PRIMARY_MAPPING}): {'PASS' if evaluation['H1_isolated_invariance']['pass'] else 'FAIL'}")
    print(f"  Median joint RMSE: {evaluation['H1_isolated_invariance']['median_joint_rmse']:.6f}")
    print(f"  Threshold: {evaluation['H1_isolated_invariance']['threshold']:.6f}")
    print(f"  Mapping: {evaluation['H1_isolated_invariance']['mapping']}")
    
    print(f"\nH2 (Reactive Breaks - {PRIMARY_MAPPING}): {'PASS' if evaluation['H2_reactive_breaks']['pass'] else 'FAIL'}")
    print(f"  Reactive median RMSE: {evaluation['H2_reactive_breaks']['median_joint_rmse']:.6f}")
    print(f"  Isolated median RMSE: {evaluation['H2_reactive_breaks']['isolated_median']:.6f}")
    print(f"  Ratio: {evaluation['H2_reactive_breaks']['ratio']:.2f}")
    print(f"  Min ratio threshold: {evaluation['H2_reactive_breaks']['min_ratio_threshold']:.2f}")
    print(f"  Mapping: {evaluation['H2_reactive_breaks']['mapping']}")
    
    # Print sensitivity analysis
    print(f"\nSENSITIVITY ANALYSIS:")
    for mapping in ['D1', 'D2']:
        sens = evaluation['sensitivity'][mapping]
        print(f"  {mapping}: Isolated={sens['isolated_median']:.6f}, Reactive={sens['reactive_median']:.6f}, Ratio={sens['ratio']:.2f}")
    
    print(f"\nOVERALL TEST: {'PASS' if evaluation['overall_pass'] else 'FAIL'}")
    
    # Save results
    save_results(results, evaluation)
    create_plots(results)
    
    print(f"\nResults saved to results/T1_summary.json and results/T1_metrics.csv")
    print(f"Plots saved to results/T1_plots.png")
    
    return evaluation['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)