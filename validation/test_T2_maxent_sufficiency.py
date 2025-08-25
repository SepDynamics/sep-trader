#!/usr/bin/env python3
"""
Test T2: Pairwise Maximum-Entropy Sufficiency
Tests H3: Pairwise joint triads achieve maximum entropy distribution
Tests H4: Adding more variables does not significantly improve entropy beyond pairwise
"""

import numpy as np
import json
import csv
import os
from pathlib import Path
from itertools import combinations
from scipy.stats import entropy
from sklearn.covariance import LedoitWolf
from sep_core import (
    triad_series, rmse, bit_mapping_D1, bit_mapping_D2,
    generate_poisson_process, generate_van_der_pol, RANDOM_SEED
)
import matplotlib.pyplot as plt

# Memory and performance parameters
SUBSAMPLE = 10     # keep 1 in 10 samples to reduce T
MAX_COND_ORDER = 2 # only test 1 and 2 conditioners

# Test parameters
PROCESS_LENGTH = 20000
BETA = 0.1  # EMA parameter
N_PROCESSES = 4  # Number of processes to test
ENTROPY_THRESHOLD = 0.3  # Minimum normalized entropy for H3 (30% reduction)
SUFFICIENCY_THRESHOLD = 0.05  # Maximum entropy gain threshold for H4 (5% relative gain)

def create_results_dir():
    """Create results directory if it doesn't exist."""
    Path("results").mkdir(exist_ok=True)

def normalize_entropy(H: float, n_bins: int) -> float:
    """Normalize entropy by theoretical maximum (uniform distribution)."""
    max_entropy = np.log2(n_bins) if n_bins > 1 else 0
    return H / max_entropy if max_entropy > 0 else 0

def gaussian_entropy(X: np.ndarray) -> float:
    """Differential entropy of multivariate Gaussian with shrinkage covariance."""
    X = np.asarray(X, dtype=np.float32)
    # Standardize input (zero mean, unit variance per channel)
    Xc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    
    if Xc.shape[0] < 2:
        return 0.0
    
    # Use Ledoit-Wolf shrinkage for numerical stability
    cov = LedoitWolf().fit(Xc).covariance_
    d = Xc.shape[1]
    
    # Ensure positive definiteness
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + 1e-6 * np.eye(d, dtype=cov.dtype)
        sign, logdet = np.linalg.slogdet(cov)
    
    # Differential entropy in nats: 0.5 * (d * ln(2πe) + ln|Σ|)
    H_nats = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
    # Convert to bits
    return H_nats / np.log(2)

def compute_triad_entropy_gaussian(triads: np.ndarray) -> dict:
    """Entropy stats via Gaussian approximation."""
    H = gaussian_entropy(triads)  # joint HCS
    # Individual components
    H_ent = gaussian_entropy(triads[:, [0]])
    C_ent = gaussian_entropy(triads[:, [1]])
    S_ent = gaussian_entropy(triads[:, [2]])
    return {
        'individual': {'H': H_ent, 'C': C_ent, 'S': S_ent},
        'triple': {'HCS': H},
    }

def generate_multi_process_data(n_processes: int, process_length: int, seed: int = RANDOM_SEED) -> list:
    """Generate multiple independent processes for multi-variable analysis."""
    np.random.seed(seed)
    processes = []
    
    for i in range(n_processes):
        if i % 2 == 0:
            # Alternate between isolated and reactive processes
            signal = generate_poisson_process(process_length, rate=1.0 + 0.2*i, seed=seed + i)
        else:
            signal = generate_van_der_pol(process_length, mu=1.0 + 0.1*i, seed=seed + i)
        processes.append(signal)
    
    return processes

def compute_multiprocess_triads(processes: list, bit_mapping: str) -> list:
    """Compute triads for multiple processes."""
    triads_list = []
    
    for signal in processes:
        if bit_mapping == "D1":
            bits = bit_mapping_D1(signal)
        elif bit_mapping == "D2":
            bits = bit_mapping_D2(signal)
        else:
            raise ValueError(f"Unknown bit mapping: {bit_mapping}")
        
        triads = triad_series(bits, beta=BETA)
        triads = triads[::SUBSAMPLE].astype(np.float32, copy=False)  # subsample + cast
        triads_list.append(triads)
    
    return triads_list

def conditional_entropy_gaussian(X: np.ndarray, Y: np.ndarray) -> float:
    XY = np.hstack([X, Y]).astype(np.float32)
    return gaussian_entropy(XY) - gaussian_entropy(Y)

def conditional_gain_gaussian(triads_list: list, target_idx: int, conditioning_indices: list) -> float:
    """
    ΔH = H(T_i | T_cond) - H(T_i | T_pair), computed relative to the best single conditioner.
    If conditioning_indices is length 1, return H(T_i | T_j).
    If length 2, return H(T_i | T_j, T_k) - min_j H(T_i | T_j).
    """
    Ti = triads_list[target_idx]
    conds = [triads_list[idx] for idx in conditioning_indices]
    if len(conditioning_indices) == 1:
        Hj = conditional_entropy_gaussian(Ti, conds[0])
        return float(Hj)
    elif len(conditioning_indices) == 2:
        Hj_best = np.inf
        for j in conditioning_indices:
            Hj = conditional_entropy_gaussian(Ti, triads_list[j])
            Hj_best = min(Hj_best, Hj)
        Hjk = conditional_entropy_gaussian(Ti, np.hstack(conds))
        return float(Hjk - Hj_best)
    else:
        raise ValueError("Only orders 1 and 2 are supported")

def run_maxent_sufficiency_test(bit_mapping: str, seed: int = RANDOM_SEED) -> dict:
    """Run maximum entropy sufficiency test."""
    print(f"Running T2 test with {bit_mapping} mapping")
    
    # Generate multi-process data
    processes = generate_multi_process_data(N_PROCESSES, PROCESS_LENGTH, seed)
    triads_list = compute_multiprocess_triads(processes, bit_mapping)
    
    results = {
        'bit_mapping': bit_mapping,
        'seed': seed,
        'n_processes': N_PROCESSES,
        'individual_entropies': [],
        'pairwise_analysis': {},
        'multivariate_analysis': {}
    }
    
    # Analyze individual process entropies
    print("  Computing individual process entropies...")
    for i, triads in enumerate(triads_list):
        entropy_data = compute_triad_entropy_gaussian(triads)
        results['individual_entropies'].append({
            'process_idx': i,
            'entropy_data': entropy_data
        })
        
        print(f"    Process {i}: Joint entropy = {entropy_data['triple']['HCS']:.4f}")
    
    # Analyze multivariate sufficiency
    print("  Testing multivariate sufficiency...")
    multivariate_gains = []
    
    for target_idx in range(N_PROCESSES):
        others = [i for i in range(N_PROCESSES) if i != target_idx]
        # order-1
        for j in others:
            gain1 = conditional_gain_gaussian(triads_list, target_idx, [j])
            multivariate_gains.append({'target': target_idx, 'conditioning': (j,), 'n_conditioning': 1, 'entropy_gain': gain1})
        # order-2
        for j, k in combinations(others, 2):
            gain2 = conditional_gain_gaussian(triads_list, target_idx, [j, k])
            multivariate_gains.append({'target': target_idx, 'conditioning': (j, k), 'n_conditioning': 2, 'entropy_gain': gain2})
    
    results['multivariate_analysis'] = multivariate_gains
    
    return results

def evaluate_hypotheses(results: list) -> dict:
    """Evaluate H3 and H4 hypotheses."""
    pair_cond = []  # H(T_i | T_j)
    order2_excess = []  # H(T_i | T_j, T_k) - min_j H(T_i | T_j)
    base_H = []

    for res in results:
        for ind in res['individual_entropies']:
            base_H.append(ind['entropy_data']['triple']['HCS'])
        for g in res['multivariate_analysis']:
            if g['n_conditioning'] == 1:
                pair_cond.append(g['entropy_gain'])
            elif g['n_conditioning'] == 2:
                order2_excess.append(max(0.0, g['entropy_gain']))  # only extra over best single

    med_base = float(np.median(base_H)) if base_H else 0.0
    med_pair_cond = float(np.median(pair_cond)) if pair_cond else 0.0
    med_excess = float(np.median(order2_excess)) if order2_excess else 0.0

    # normalize pair conditional relative to base
    pair_reduction = (med_base - med_pair_cond) / max(1e-9, med_base)
    h3_pass = pair_reduction >= 0.3  # e.g., ≥30% reduction by single conditioner

    # normalize order-2 excess relative to base entropy
    norm_excess = med_excess / max(1e-9, med_base)
    h4_pass = norm_excess <= 0.05  # ≤5% relative gain

    return {
        'H3_maxent_pairwise': {
            'median_base_entropy': med_base,
            'median_pair_cond_entropy': med_pair_cond,
            'relative_reduction': pair_reduction,
            'reduction_threshold': 0.30,
            'pass': bool(h3_pass),
        },
        'H4_pairwise_sufficiency': {
            'median_order2_excess': med_excess,
            'normalized_excess': norm_excess,
            'threshold': 0.05,
            'pass': bool(h4_pass),
        },
        'overall_pass': bool(h3_pass and h4_pass),
    }

def convert_to_native_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
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
        'test': 'T2_maxent_sufficiency',
        'parameters': {
            'process_length': int(PROCESS_LENGTH),
            'beta': float(BETA),
            'n_processes': int(N_PROCESSES),
            'entropy_threshold': float(ENTROPY_THRESHOLD),
            'sufficiency_threshold': float(SUFFICIENCY_THRESHOLD),
            'subsample': int(SUBSAMPLE)
        },
        'results': results_clean,
        'evaluation': evaluation_clean
    }
    
    with open('results/T2_summary.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save entropy metrics to CSV
    with open('results/T2_entropy_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bit_mapping', 'seed', 'process_idx', 'H_entropy', 'C_entropy', 'S_entropy', 'HCS_entropy'])
        
        for res in results:
            for ind_data in res['individual_entropies']:
                entropy_data = ind_data['entropy_data']
                writer.writerow([
                    res['bit_mapping'],
                    res['seed'],
                    ind_data['process_idx'],
                    entropy_data['individual']['H'],
                    entropy_data['individual']['C'],
                    entropy_data['individual']['S'],
                    entropy_data['triple']['HCS']
                ])

def create_plots(results: list, evaluation: dict):
    """Create visualization plots."""
    create_results_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('T2: Pairwise Maximum-Entropy Sufficiency Results')
    
    # Collect entropy data
    individual_entropies = []
    pair_cond_entropies = []
    order2_excess_gains = []
    
    for res in results:
        for ind_data in res['individual_entropies']:
            individual_entropies.append(ind_data['entropy_data']['triple']['HCS'])
        
        for gain_data in res['multivariate_analysis']:
            if gain_data['n_conditioning'] == 1:
                pair_cond_entropies.append(gain_data['entropy_gain'])
            elif gain_data['n_conditioning'] == 2:
                order2_excess_gains.append(max(0.0, gain_data['entropy_gain']))
    
    # Plot base entropy distribution
    ax = axes[0, 0]
    ax.hist(individual_entropies, bins=20, alpha=0.7, label='Base H(T_i)', density=True)
    ax.set_xlabel('Entropy H(T_i)')
    ax.set_ylabel('Density')
    ax.set_title('Base Entropy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot conditional entropy distribution
    ax = axes[0, 1]
    ax.hist(pair_cond_entropies, bins=20, alpha=0.7, label='H(T_i|T_j)', density=True)
    ax.set_xlabel('Conditional Entropy H(T_i|T_j)')
    ax.set_ylabel('Density')
    ax.set_title('Pairwise Conditional Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot entropy reduction
    ax = axes[1, 0]
    if individual_entropies and pair_cond_entropies:
        base_med = np.median(individual_entropies)
        cond_med = np.median(pair_cond_entropies)
        reduction = (base_med - cond_med) / max(1e-9, base_med)
        ax.bar(['Reduction'], [reduction], alpha=0.7)
        ax.axhline(y=0.3, color='r', linestyle='--', label='Threshold (0.3)')
        ax.set_ylabel('Relative Entropy Reduction')
        ax.set_title('Pairwise Sufficiency (H3)')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot order-2 excess gains
    ax = axes[1, 1]
    ax.hist(order2_excess_gains, bins=20, alpha=0.7, label='Order-2 Excess', density=True)
    ax.axvline(x=SUFFICIENCY_THRESHOLD, color='r', linestyle='--',
               label=f'Threshold ({SUFFICIENCY_THRESHOLD})')
    ax.set_xlabel('Excess Gain H(T_i|T_j,T_k) - min H(T_i|T_j)')
    ax.set_ylabel('Density')
    ax.set_title('Higher-Order Excess (H4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/T2_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the complete T2 test suite."""
    print("="*60)
    print("Running Test T2: Pairwise Maximum-Entropy Sufficiency")
    print("="*60)
    
    results = []
    
    # Test both bit mappings
    for bit_mapping in ["D1", "D2"]:
        result = run_maxent_sufficiency_test(bit_mapping, seed=RANDOM_SEED)
        results.append(result)
    
    # Evaluate hypotheses
    evaluation = evaluate_hypotheses(results)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"H3 (Maximum Entropy Pairwise): {'PASS' if evaluation['H3_maxent_pairwise']['pass'] else 'FAIL'}")
    print(f"  Median base entropy: {evaluation['H3_maxent_pairwise']['median_base_entropy']:.4f}")
    print(f"  Median pair conditional entropy: {evaluation['H3_maxent_pairwise']['median_pair_cond_entropy']:.4f}")
    print(f"  Relative reduction: {evaluation['H3_maxent_pairwise']['relative_reduction']:.4f}")
    print(f"  Threshold: {evaluation['H3_maxent_pairwise']['reduction_threshold']:.4f}")
    
    print(f"\nH4 (Pairwise Sufficiency): {'PASS' if evaluation['H4_pairwise_sufficiency']['pass'] else 'FAIL'}")
    print(f"  Median order-2 excess: {evaluation['H4_pairwise_sufficiency']['median_order2_excess']:.4f}")
    print(f"  Normalized excess: {evaluation['H4_pairwise_sufficiency']['normalized_excess']:.4f}")
    print(f"  Threshold: {evaluation['H4_pairwise_sufficiency']['threshold']:.4f}")
    
    print(f"\nOVERALL TEST: {'PASS' if evaluation['overall_pass'] else 'FAIL'}")
    
    # Save results
    save_results(results, evaluation)
    create_plots(results, evaluation)
    
    print(f"\nResults saved to results/T2_summary.json and results/T2_entropy_metrics.csv")
    print(f"Plots saved to results/T2_plots.png")
    
    return evaluation['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)