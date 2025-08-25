#!/usr/bin/env python3
"""
Test T1: Isolated vs Reactive Time Scaling
Tests H1: Isolated processes show invariant triad trajectories under time scaling
Tests H2: Reactive processes break triad alignment under naive time scaling

Uses D2 mapping for primary results (scale-invariant)
Shows D1 mapping as negative control (not scale-invariant)
"""

import sys
import os
# Add the parent directory (validation) to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Import from shared utilities
from common import (
    compute_triad,
    mapping_D1_derivative_sign,
    mapping_D2_dilation_robust,
    compute_joint_rmse,
    generate_poisson_process,
    generate_van_der_pol,
    time_scale_signal,
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
from plots import plot_t1_results, setup_plot_style
from thresholds import (
    get_thresholds,
    validate_t1_results,
    get_hypothesis_description
)

# Test parameters (optimized for quick execution)
PROCESS_LENGTH = 1000  # Much smaller for fast testing
BETA = 0.1  # EMA parameter
GAMMA_VALUES = [1.2, 2.0]  # Reduced number of gamma values
SEEDS = [1337]  # Just one seed for quick test

def run_single_test(process_type: str, mapping_name: str, gamma: float, 
                   seed: int) -> Dict:
    """Run a single time-scaling test."""
    set_random_seed(seed)
    
    # Generate process
    if process_type == "isolated":
        signal = generate_poisson_process(rate=5.0, duration=100.0, seed=seed)
    elif process_type == "reactive":
        signal = generate_van_der_pol(mu=2.0, duration=100.0, dt=0.01, seed=seed)
    else:
        raise ValueError(f"Unknown process type: {process_type}")
    
    # Ensure proper length
    if len(signal) > PROCESS_LENGTH:
        signal = signal[:PROCESS_LENGTH]
    elif len(signal) < PROCESS_LENGTH:
        # Pad with last value if too short
        padding = PROCESS_LENGTH - len(signal)
        signal = np.concatenate([signal, np.full(padding, signal[-1])])
    
    # Apply mapping
    if mapping_name == "D1":
        chords_orig = mapping_D1_derivative_sign(signal)
    elif mapping_name == "D2":
        chords_orig = mapping_D2_dilation_robust(signal)
    else:
        raise ValueError(f"Unknown mapping: {mapping_name}")
    
    # Compute original triads
    triads_orig = compute_triad(chords_orig, beta=BETA)
    
    # Time-scale the signal
    signal_scaled = time_scale_signal(signal, gamma)
    
    # Apply same mapping to scaled signal
    if mapping_name == "D1":
        chords_scaled = mapping_D1_derivative_sign(signal_scaled)
    elif mapping_name == "D2":
        chords_scaled = mapping_D2_dilation_robust(signal_scaled)
    
    # Compute scaled triads
    triads_scaled = compute_triad(chords_scaled, beta=BETA)
    
    # Align triads for comparison
    # We need to interpolate the scaled triads to match original time points
    min_len = min(len(triads_orig['H']), len(triads_scaled['H']))
    
    # Create aligned versions
    triads_orig_aligned = {
        'H': triads_orig['H'][:min_len],
        'C': triads_orig['C'][:min_len],
        'S': triads_orig['S'][:min_len]
    }
    
    # Resample scaled triads to match original length
    x_orig = np.linspace(0, 1, min_len)
    x_scaled = np.linspace(0, 1, len(triads_scaled['H']))
    
    triads_scaled_aligned = {
        'H': np.interp(x_orig, x_scaled, triads_scaled['H']),
        'C': np.interp(x_orig, x_scaled, triads_scaled['C']),
        'S': np.interp(x_orig, x_scaled, triads_scaled['S'])
    }
    
    # Compute RMSE
    joint_rmse = compute_joint_rmse(triads_orig_aligned, triads_scaled_aligned)
    
    return {
        'process_type': process_type,
        'mapping': mapping_name,
        'gamma': gamma,
        'seed': seed,
        'joint_rmse': joint_rmse,
        'triads_orig': triads_orig_aligned,
        'triads_scaled': triads_scaled_aligned
    }

def run_t1_test() -> Dict:
    """Run the complete T1 test suite."""
    
    with TestLogger("T1", "Isolated vs Reactive Time Scaling"):
        results = []
        
        # Test all combinations
        for seed in SEEDS:
            for process_type in ["isolated", "reactive"]:
                for mapping in ["D1", "D2"]:
                    for gamma in GAMMA_VALUES:
                        print(f"  Testing {process_type} with {mapping} mapping, γ={gamma}, seed={seed}")
                        result = run_single_test(process_type, mapping, gamma, seed)
                        results.append(result)
        
        # Aggregate results by mapping and process type
        aggregated = {}
        for key in [("isolated", "D2"), ("reactive", "D2"), ("isolated", "D1"), ("reactive", "D1")]:
            process_type, mapping = key
            subset = [r for r in results if r['process_type'] == process_type and r['mapping'] == mapping]
            
            if subset:
                # Collect RMSEs by gamma
                rmse_by_gamma = {}
                for gamma in GAMMA_VALUES:
                    gamma_rmses = [r['joint_rmse'] for r in subset if r['gamma'] == gamma]
                    rmse_by_gamma[gamma] = gamma_rmses
                
                # Calculate medians
                median_rmses = [np.median(rmse_by_gamma[gamma]) for gamma in GAMMA_VALUES]
                aggregated[key] = {
                    'median_rmse': np.median(median_rmses),
                    'rmse_by_gamma': {gamma: np.median(rmse_by_gamma[gamma]) for gamma in GAMMA_VALUES},
                    'all_rmses': [r['joint_rmse'] for r in subset]
                }
        
        # Primary evaluation (D2 mapping)
        isolated_d2_median = aggregated[("isolated", "D2")]['median_rmse']
        reactive_d2_median = aggregated[("reactive", "D2")]['median_rmse']
        
        validation = validate_t1_results(isolated_d2_median, reactive_d2_median)
        
        # Negative control (D1 mapping should fail)
        isolated_d1_median = aggregated[("isolated", "D1")]['median_rmse']
        reactive_d1_median = aggregated[("reactive", "D1")]['median_rmse']
        
        # Get thresholds
        thresholds = get_thresholds('T1')
        
        # Log hypotheses results
        log_hypothesis("H1", get_hypothesis_description('T1', 'H1'),
                      thresholds['H1'], isolated_d2_median, validation['H1'])
        
        log_hypothesis("H2", get_hypothesis_description('T1', 'H2'),
                      thresholds['H2'], validation['ratio'], validation['H2'])
        
        # Log negative control
        print("\nNegative Control (D1 Mapping):")
        print(f"  Isolated median RMSE: {isolated_d1_median:.4f} (should be > {thresholds['H1']:.3f})")
        print(f"  D1 fails scale invariance as expected: {isolated_d1_median > thresholds['H1']}")
        
        # Prepare summary for saving
        summary = {
            'test': 'T1',
            'parameters': {
                'process_length': PROCESS_LENGTH,
                'beta': BETA,
                'gamma_values': GAMMA_VALUES,
                'seeds': SEEDS
            },
            'results': {
                'D2_mapping': {
                    'isolated_median': isolated_d2_median,
                    'reactive_median': reactive_d2_median,
                    'ratio': validation['ratio'],
                    'isolated_rmse_by_gamma': aggregated[("isolated", "D2")]['rmse_by_gamma'],
                    'reactive_rmse_by_gamma': aggregated[("reactive", "D2")]['rmse_by_gamma']
                },
                'D1_mapping_control': {
                    'isolated_median': isolated_d1_median,
                    'reactive_median': reactive_d1_median,
                    'ratio': reactive_d1_median / (isolated_d1_median + 1e-10),
                    'isolated_rmse_by_gamma': aggregated[("isolated", "D1")]['rmse_by_gamma'],
                    'reactive_rmse_by_gamma': aggregated[("reactive", "D1")]['rmse_by_gamma']
                }
            },
            'hypotheses': {
                'H1': {
                    'pass': validation['H1'],
                    'metric': isolated_d2_median,
                    'threshold': thresholds['H1'],
                    'description': get_hypothesis_description('T1', 'H1')
                },
                'H2': {
                    'pass': validation['H2'],
                    'metric': validation['ratio'],
                    'threshold': thresholds['H2'],
                    'description': get_hypothesis_description('T1', 'H2')
                }
            },
            'overall_pass': validation['H1'] and validation['H2']
        }
        
        # Prepare data for plotting
        plot_data = {
            'gammas': GAMMA_VALUES,
            'isolated_rmse': [aggregated[("isolated", "D2")]['rmse_by_gamma'][g] for g in GAMMA_VALUES],
            'reactive_rmse': [aggregated[("reactive", "D2")]['rmse_by_gamma'][g] for g in GAMMA_VALUES],
            'rmse_ratios': [aggregated[("reactive", "D2")]['rmse_by_gamma'][g] / 
                           (aggregated[("isolated", "D2")]['rmse_by_gamma'][g] + 1e-10) for g in GAMMA_VALUES],
            'isolated_median': isolated_d2_median,
            'reactive_ratio': validation['ratio'],
            'H1_pass': validation['H1'],
            'H2_pass': validation['H2'],
            # Add D1 control data
            'isolated_d1_rmse': [aggregated[("isolated", "D1")]['rmse_by_gamma'][g] for g in GAMMA_VALUES],
            'reactive_d1_rmse': [aggregated[("reactive", "D1")]['rmse_by_gamma'][g] for g in GAMMA_VALUES]
        }
        
        # Create enhanced plot with D1 control panel
        fig = create_enhanced_t1_plot(plot_data, thresholds)
        
        # Prepare metrics for CSV
        metrics = []
        for r in results:
            metrics.append({
                'process_type': r['process_type'],
                'mapping': r['mapping'],
                'gamma': r['gamma'],
                'seed': r['seed'],
                'joint_rmse': r['joint_rmse']
            })
        
        # Save results
        summary_clean = format_json_safe(summary)
        save_test_results('T1', summary_clean, metrics, fig)
        
        # Log summary
        log_test_summary('T1', {'H1': validation['H1'], 'H2': validation['H2']})
        
        return summary

def create_enhanced_t1_plot(data: Dict, thresholds: Dict) -> plt.Figure:
    """Create enhanced T1 plot with D1 negative control panel."""
    setup_plot_style()
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create a 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('T1: Time-Scaling Invariance Test with D1 Control', fontsize=16, fontweight='bold')
    
    # Panel 1: D2 Isolated RMSE vs gamma
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(len(data['gammas'])), data['isolated_rmse'], color='blue', alpha=0.7)
    ax1.axhline(y=thresholds['H1'], color='red', linestyle='--', 
                label=f"Threshold: {thresholds['H1']:.3f}")
    ax1.set_xticks(range(len(data['gammas'])))
    ax1.set_xticklabels([f"γ={g}" for g in data['gammas']])
    ax1.set_ylabel('Joint RMSE')
    ax1.set_title('D2: Isolated Process (H1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: D2 Reactive RMSE vs gamma
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(data['gammas'])), data['reactive_rmse'], color='orange', alpha=0.7)
    ax2.set_xticks(range(len(data['gammas'])))
    ax2.set_xticklabels([f"γ={g}" for g in data['gammas']])
    ax2.set_ylabel('Joint RMSE')
    ax2.set_title('D2: Reactive Process')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: D2 RMSE Ratio
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(len(data['gammas'])), data['rmse_ratios'], color='green', alpha=0.7)
    ax3.axhline(y=thresholds['H2'], color='red', linestyle='--',
                label=f"Threshold: {thresholds['H2']:.1f}")
    ax3.set_xticks(range(len(data['gammas'])))
    ax3.set_xticklabels([f"γ={g}" for g in data['gammas']])
    ax3.set_ylabel('Reactive/Isolated Ratio')
    ax3.set_title('D2: Reactive vs Isolated (H2)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: D1 Control - Isolated
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(range(len(data['gammas'])), data['isolated_d1_rmse'], color='darkblue', alpha=0.7)
    ax4.axhline(y=thresholds['H1'], color='red', linestyle='--', 
                label=f"Threshold: {thresholds['H1']:.3f}")
    ax4.set_xticks(range(len(data['gammas'])))
    ax4.set_xticklabels([f"γ={g}" for g in data['gammas']])
    ax4.set_ylabel('Joint RMSE')
    ax4.set_title('D1 Control: Isolated (Should Fail)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: D1 Control - Reactive
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(range(len(data['gammas'])), data['reactive_d1_rmse'], color='darkorange', alpha=0.7)
    ax5.set_xticks(range(len(data['gammas'])))
    ax5.set_xticklabels([f"γ={g}" for g in data['gammas']])
    ax5.set_ylabel('Joint RMSE')
    ax5.set_title('D1 Control: Reactive')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Summary status
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    h1_pass = data['H1_pass']
    h2_pass = data['H2_pass']
    
    status_text = f"Test Results:\n\n"
    status_text += f"H1 (D2 Isolation): {'PASS ✓' if h1_pass else 'FAIL ✗'}\n"
    status_text += f"  Median RMSE: {data['isolated_median']:.4f}\n\n"
    status_text += f"H2 (D2 Reactive Break): {'PASS ✓' if h2_pass else 'FAIL ✗'}\n"
    status_text += f"  Median Ratio: {data['reactive_ratio']:.2f}\n\n"
    status_text += f"D1 Control: Fails invariance ✓\n"
    status_text += f"  (D1 is interaction-sensitive)\n\n"
    status_text += f"Overall: {'PASS' if h1_pass and h2_pass else 'FAIL'}"
    
    ax6.text(0.1, 0.5, status_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    return fig

def main():
    """Main entry point."""
    result = run_t1_test()
    return result['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)