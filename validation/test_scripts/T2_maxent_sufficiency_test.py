#!/usr/bin/env python3
"""
Test T2: Pairwise Maximum-Entropy Sufficiency
Tests H3: Pairwise conditioning captures significant information (≥30% at ρ≥0.6)
Tests H4: Higher-order terms contribute negligible information (≤5%)

Uses D1 mapping for interaction sensitivity
Tests across ρ values [0.0, 0.2, 0.4, 0.6, 0.8] to show monotonic information capture
"""

import sys
import os
# Add the parent directory (validation) to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import from shared utilities
from common import (
    compute_triad,
    mapping_D1_derivative_sign,
    mapping_D2_dilation_robust,
    gaussian_entropy_bits,
    generate_latent_coupled_processes,
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
from plots import plot_t2_results, setup_plot_style
from thresholds import (
    get_thresholds,
    validate_t2_results,
    get_hypothesis_description
)

# Test parameters
PROCESS_LENGTH = 10000  # Length of each process
N_PROCESSES = 4  # Number of processes
BETA = 0.1  # EMA parameter for triad computation
RHO_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]  # Correlation values to test
SEEDS = [1337, 1729, 2718]  # Random seeds for robustness
SUBSAMPLE_RATE = 10  # Subsample to reduce memory usage

def compute_conditional_entropies(triads: List[Dict], subsample: int = 1) -> Dict:
    """
    Compute conditional entropies for triads.
    
    Returns:
        Dictionary with entropy metrics
    """
    # Stack triads into matrices
    n_processes = len(triads)
    n_samples = len(triads[0]['H']) // subsample
    
    # Create feature matrices (subsampled)
    features = []
    for i in range(n_processes):
        triad_matrix = np.column_stack([
            triads[i]['H'][::subsample][:n_samples],
            triads[i]['C'][::subsample][:n_samples],
            triads[i]['S'][::subsample][:n_samples]
        ])
        features.append(triad_matrix)
    
    results = {
        'marginal': [],
        'pairwise_conditional': [],
        'order2_conditional': [],
        'best_pair': [],
        'relative_reduction': [],
        'order2_excess': []
    }
    
    # For each target process
    for i in range(n_processes):
        # Marginal entropy H(Ti)
        h_marginal = gaussian_entropy_bits(features[i])
        results['marginal'].append(h_marginal)
        
        # Pairwise conditional entropies H(Ti|Tj)
        pairwise = []
        for j in range(n_processes):
            if i != j:
                # Joint features
                joint = np.hstack([features[i], features[j]])
                h_joint = gaussian_entropy_bits(joint)
                h_j = gaussian_entropy_bits(features[j])
                
                # Conditional entropy using chain rule: H(Ti|Tj) = H(Ti,Tj) - H(Tj)
                h_conditional = h_joint - h_j
                h_conditional = max(0, h_conditional)  # Ensure non-negative
                pairwise.append(h_conditional)
        
        # Best pairwise (minimum conditional entropy)
        if pairwise:
            best_pairwise = min(pairwise)
            results['pairwise_conditional'].append(pairwise)
            results['best_pair'].append(best_pairwise)
            
            # Relative reduction: [H(Ti) - H(Ti|Tj*)] / H(Ti)
            reduction = (h_marginal - best_pairwise) / (h_marginal + 1e-10)
            results['relative_reduction'].append(max(0, reduction))
        else:
            results['best_pair'].append(h_marginal)
            results['relative_reduction'].append(0.0)
        
        # Order-2 conditional entropies H(Ti|Tj,Tk)
        order2 = []
        for j in range(n_processes):
            if i == j:
                continue
            for k in range(j+1, n_processes):
                if i == k:
                    continue
                
                # Three-way joint
                joint3 = np.hstack([features[i], features[j], features[k]])
                h_joint3 = gaussian_entropy_bits(joint3)
                
                # Joint of conditions
                joint_jk = np.hstack([features[j], features[k]])
                h_jk = gaussian_entropy_bits(joint_jk)
                
                # Conditional entropy: H(Ti|Tj,Tk) = H(Ti,Tj,Tk) - H(Tj,Tk)
                h_conditional2 = h_joint3 - h_jk
                h_conditional2 = max(0, h_conditional2)
                order2.append(h_conditional2)
        
        if order2:
            best_order2 = min(order2)
            results['order2_conditional'].append(order2)
            
            # Order-2 excess: [H(Ti|Tj*) - H(Ti|Tj*,Tk*)] / H(Ti)
            # This measures additional information from second predictor
            if pairwise:
                excess = (best_pairwise - best_order2) / (h_marginal + 1e-10)
                results['order2_excess'].append(max(0, excess))
            else:
                results['order2_excess'].append(0.0)
        else:
            results['order2_excess'].append(0.0)
    
    return results

def run_single_rho_test(rho: float, seed: int) -> Dict:
    """Run test for a single correlation value."""
    set_random_seed(seed)
    
    print(f"    Testing ρ={rho:.1f}, seed={seed}")
    
    # Generate coupled processes
    processes = generate_latent_coupled_processes(
        n_processes=N_PROCESSES,
        rho=rho,
        length=PROCESS_LENGTH,
        seed=seed
    )
    
    # Apply D1 mapping (interaction-sensitive)
    triads = []
    for process in processes:
        chords = mapping_D1_derivative_sign(process)
        triad = compute_triad(chords, beta=BETA)
        triads.append(triad)
    
    # Compute conditional entropies
    entropy_results = compute_conditional_entropies(triads, subsample=SUBSAMPLE_RATE)
    
    # Calculate medians
    median_reduction = np.median(entropy_results['relative_reduction'])
    median_excess = np.median(entropy_results['order2_excess'])
    
    return {
        'rho': rho,
        'seed': seed,
        'median_reduction': median_reduction,
        'median_excess': median_excess,
        'all_reductions': entropy_results['relative_reduction'],
        'all_excess': entropy_results['order2_excess'],
        'marginal_entropies': entropy_results['marginal']
    }

def run_t2_test() -> Dict:
    """Run the complete T2 test suite."""
    
    with TestLogger("T2", "Pairwise Maximum-Entropy Sufficiency"):
        all_results = []
        
        # Test each rho value
        for rho in RHO_VALUES:
            print(f"  Testing correlation ρ={rho:.1f}")
            rho_results = []
            
            for seed in SEEDS:
                result = run_single_rho_test(rho, seed)
                rho_results.append(result)
                all_results.append(result)
            
            # Aggregate for this rho
            median_reduction = np.median([r['median_reduction'] for r in rho_results])
            median_excess = np.median([r['median_excess'] for r in rho_results])
            
            print(f"    Median reduction: {median_reduction:.3f}")
            print(f"    Median excess: {median_excess:.4f}")
        
        # Aggregate results by rho
        reductions_by_rho = {}
        excess_by_rho = {}
        
        for rho in RHO_VALUES:
            rho_subset = [r for r in all_results if r['rho'] == rho]
            
            # Collect all individual reductions and excesses
            all_reductions = []
            all_excess = []
            for r in rho_subset:
                all_reductions.extend(r['all_reductions'])
                all_excess.extend(r['all_excess'])
            
            reductions_by_rho[rho] = np.median(all_reductions) if all_reductions else 0
            excess_by_rho[rho] = np.median(all_excess) if all_excess else 0
        
        # Validate hypotheses
        overall_median_excess = np.median([r['median_excess'] for r in all_results])
        validation = validate_t2_results(reductions_by_rho, overall_median_excess)
        
        # Get thresholds
        thresholds = get_thresholds('T2')
        
        # Log hypothesis results
        log_hypothesis("H3", get_hypothesis_description('T2', 'H3'),
                      thresholds['H3'], validation['max_reduction'], validation['H3'])
        
        log_hypothesis("H4", get_hypothesis_description('T2', 'H4'),
                      thresholds['H4'], overall_median_excess, validation['H4'])
        
        # Show rho-dependent results
        print("\nReduction by correlation:")
        for rho in RHO_VALUES:
            print(f"  ρ={rho:.1f}: {reductions_by_rho[rho]:.3f}")
        
        # Prepare summary
        summary = {
            'test': 'T2',
            'parameters': {
                'process_length': PROCESS_LENGTH,
                'n_processes': N_PROCESSES,
                'beta': BETA,
                'rho_values': RHO_VALUES,
                'seeds': SEEDS,
                'subsample_rate': SUBSAMPLE_RATE,
                'mapping': 'D1'
            },
            'results': {
                'reductions_by_rho': reductions_by_rho,
                'excess_by_rho': excess_by_rho,
                'overall_median_excess': overall_median_excess,
                'max_reduction': validation['max_reduction']
            },
            'hypotheses': {
                'H3': {
                    'pass': validation['H3'],
                    'metric': validation['max_reduction'],
                    'threshold': thresholds['H3'],
                    'rho_threshold': thresholds['rho_threshold'],
                    'description': get_hypothesis_description('T2', 'H3')
                },
                'H4': {
                    'pass': validation['H4'],
                    'metric': overall_median_excess,
                    'threshold': thresholds['H4'],
                    'description': get_hypothesis_description('T2', 'H4')
                }
            },
            'overall_pass': validation['H3'] and validation['H4']
        }
        
        # Prepare plot data
        plot_data = {
            'rhos': RHO_VALUES,
            'reductions': [reductions_by_rho[rho] for rho in RHO_VALUES],
            'order2_excess': [r['median_excess'] for r in all_results],
            'max_reduction': validation['max_reduction'],
            'median_excess': overall_median_excess,
            'H3_pass': validation['H3'],
            'H4_pass': validation['H4']
        }
        
        # Create plot
        fig = plot_t2_results(plot_data, thresholds)
        
        # Prepare metrics for CSV
        metrics = []
        for r in all_results:
            metrics.append({
                'rho': r['rho'],
                'seed': r['seed'],
                'median_reduction': r['median_reduction'],
                'median_excess': r['median_excess'],
                'mean_marginal_entropy': np.mean(r['marginal_entropies'])
            })
        
        # Save results
        summary_clean = format_json_safe(summary)
        save_test_results('T2', summary_clean, metrics, fig)
        
        # Log summary
        log_test_summary('T2', {'H3': validation['H3'], 'H4': validation['H4']})
        
        # Additional diagnostics
        print("\nDiagnostics:")
        print(f"  Mapping used: D1 (interaction-sensitive)")
        print(f"  Best reduction achieved: {validation['max_reduction']:.3f} at ρ≥{thresholds['rho_threshold']}")
        print(f"  Order-2 excess: {overall_median_excess:.4f} (threshold: {thresholds['H4']})")
        
        return summary

def main():
    """Main entry point."""
    result = run_t2_test()
    return result['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)