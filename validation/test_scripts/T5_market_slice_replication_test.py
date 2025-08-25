#!/usr/bin/env python3
"""
Test T5: Market Slice Replication
Tests that the invariances and pairwise behaviors survive on real price streams.

Uses simulated FX data (EURUSD, GBPUSD) with USD common driver to test:
- Time-scale invariance like T1
- Pairwise reduction like T2 (USD common driver)
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
    time_scale_signal,
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
from plots import setup_plot_style
from thresholds import (
    get_thresholds,
    validate_t5_results,
    get_hypothesis_description
)

# Test parameters (optimized for quick execution)
SAMPLE_LENGTH = 5000
BETA = 0.1
GAMMA_VALUES = [1.2, 2.0]  # Time scaling factors
SEEDS = [1337]
USD_STRENGTH_CORRELATION = 0.7  # Correlation with USD strength

def generate_fx_data(length: int, seed: int) -> Dict[str, np.ndarray]:
    """Generate realistic FX data with USD common driver."""
    set_random_seed(seed)
    
    # Generate USD strength index (common driver)
    usd_strength = np.zeros(length)
    usd_strength[0] = 100.0
    
    # USD follows mean-reverting process with trend
    for t in range(1, length):
        # Mean reversion to 100 with small trend
        mean_revert = -0.001 * (usd_strength[t-1] - 100.0)
        trend = 0.0002  # Slight upward trend
        noise = np.random.randn() * 0.5
        
        change = mean_revert + trend + noise
        usd_strength[t] = usd_strength[t-1] + change
    
    # Generate EURUSD (inverse correlation with USD strength)
    eurusd = np.zeros(length)
    eurusd[0] = 1.1000  # Starting rate
    
    for t in range(1, length):
        # Inverse correlation with USD strength
        usd_effect = -USD_STRENGTH_CORRELATION * (usd_strength[t] - usd_strength[t-1]) * 0.01
        # Idiosyncratic EUR movement
        eur_idio = np.random.randn() * 0.3
        # Combination
        change = usd_effect + eur_idio * 0.001
        eurusd[t] = eurusd[t-1] + change
    
    # Generate GBPUSD (also inverse correlation with USD strength)
    gbpusd = np.zeros(length)
    gbpusd[0] = 1.2500  # Starting rate
    
    for t in range(1, length):
        # Inverse correlation with USD strength
        usd_effect = -USD_STRENGTH_CORRELATION * (usd_strength[t] - usd_strength[t-1]) * 0.012
        # Idiosyncratic GBP movement  
        gbp_idio = np.random.randn() * 0.4
        # Combination
        change = usd_effect + gbp_idio * 0.001
        gbpusd[t] = gbpusd[t-1] + change
    
    return {
        'EURUSD': eurusd,
        'GBPUSD': gbpusd,
        'USD_INDEX': usd_strength
    }

def test_fx_time_invariance(fx_data: Dict[str, np.ndarray], gamma: float) -> Dict:
    """Test time-scale invariance on FX data."""
    results = {}
    
    for pair_name, prices in fx_data.items():
        if pair_name == 'USD_INDEX':
            continue
            
        # Apply D2 mapping to original
        chords_orig = mapping_D2_dilation_robust(prices)
        triads_orig = compute_triad(chords_orig, beta=BETA)
        
        # Time-scale the prices
        prices_scaled = time_scale_signal(prices, gamma)
        
        # Apply D2 mapping to scaled
        chords_scaled = mapping_D2_dilation_robust(prices_scaled)
        triads_scaled = compute_triad(chords_scaled, beta=BETA)
        
        # Ensure both triads have same length for comparison
        min_len = min(len(triads_orig['H']), len(triads_scaled['H']))
        
        # Truncate both triads to minimum length
        triads_orig_trunc = {
            'H': triads_orig['H'][:min_len],
            'C': triads_orig['C'][:min_len],
            'S': triads_orig['S'][:min_len]
        }
        triads_scaled_trunc = {
            'H': triads_scaled['H'][:min_len],
            'C': triads_scaled['C'][:min_len],
            'S': triads_scaled['S'][:min_len]
        }
        
        # Compute RMSE on aligned data
        joint_rmse = compute_joint_rmse(triads_orig_trunc, triads_scaled_trunc)
        
        results[pair_name] = {
            'gamma': gamma,
            'joint_rmse': joint_rmse,
            'triads_orig': triads_orig,
            'triads_scaled': triads_scaled
        }
    
    return results

def test_fx_pairwise_reduction(fx_data: Dict[str, np.ndarray]) -> Dict:
    """Test pairwise entropy reduction on FX data."""
    # Use D1 mapping for interaction sensitivity
    pair_names = ['EURUSD', 'GBPUSD']
    triads = {}
    
    for name in pair_names:
        chords = mapping_D1_derivative_sign(fx_data[name])
        triads[name] = compute_triad(chords, beta=BETA)
    
    # Create feature matrices (subsampled for speed)
    subsample = 5
    features = {}
    n_samples = len(triads['EURUSD']['H']) // subsample
    
    for name in pair_names:
        triad_matrix = np.column_stack([
            triads[name]['H'][::subsample][:n_samples],
            triads[name]['C'][::subsample][:n_samples],
            triads[name]['S'][::subsample][:n_samples]
        ])
        features[name] = triad_matrix
    
    results = {}
    
    # Test EURUSD conditioning on GBPUSD
    target = 'EURUSD'
    predictor = 'GBPUSD'
    
    # Marginal entropy H(EURUSD)
    h_marginal = gaussian_entropy_bits(features[target])
    
    # Joint entropy H(EURUSD, GBPUSD)
    joint_features = np.hstack([features[target], features[predictor]])
    h_joint = gaussian_entropy_bits(joint_features)
    
    # Predictor entropy H(GBPUSD)
    h_predictor = gaussian_entropy_bits(features[predictor])
    
    # Conditional entropy H(EURUSD|GBPUSD) = H(EURUSD,GBPUSD) - H(GBPUSD)
    h_conditional = h_joint - h_predictor
    h_conditional = max(0, h_conditional)
    
    # Relative reduction
    reduction = (h_marginal - h_conditional) / (h_marginal + 1e-10)
    
    results['target'] = target
    results['predictor'] = predictor
    results['marginal_entropy'] = h_marginal
    results['conditional_entropy'] = h_conditional
    results['entropy_reduction'] = reduction
    results['joint_entropy'] = h_joint
    results['predictor_entropy'] = h_predictor
    
    return results

def run_t5_test() -> Dict:
    """Run the complete T5 test suite."""
    
    with TestLogger("T5", "Market Slice Replication"):
        
        # Generate FX data
        print("  Generating synthetic FX data...")
        fx_data = generate_fx_data(SAMPLE_LENGTH, SEEDS[0])
        
        # Test 1: Time-scale invariance
        print("  Testing time-scale invariance on FX data...")
        invariance_results = {}
        invariance_rmses = []
        
        for gamma in GAMMA_VALUES:
            print(f"    Testing γ={gamma}")
            gamma_results = test_fx_time_invariance(fx_data, gamma)
            invariance_results[gamma] = gamma_results
            
            # Collect RMSEs
            for pair, data in gamma_results.items():
                invariance_rmses.append(data['joint_rmse'])
                print(f"      {pair}: RMSE = {data['joint_rmse']:.4f}")
        
        median_invariance_rmse = np.median(invariance_rmses)
        
        # Test 2: Pairwise entropy reduction
        print("  Testing pairwise entropy reduction...")
        reduction_results = test_fx_pairwise_reduction(fx_data)
        entropy_reduction = reduction_results['entropy_reduction']
        
        print(f"    {reduction_results['target']} | {reduction_results['predictor']}")
        print(f"    Marginal entropy: {reduction_results['marginal_entropy']:.3f}")
        print(f"    Conditional entropy: {reduction_results['conditional_entropy']:.3f}")
        print(f"    Entropy reduction: {entropy_reduction:.3f}")
        
        # Validate results
        validation = validate_t5_results(median_invariance_rmse, entropy_reduction)
        
        # Get thresholds
        thresholds = get_thresholds('T5')
        
        # Log hypothesis results
        log_hypothesis("T5 Invariance", get_hypothesis_description('T5', 'T5_invariance'),
                      thresholds['T5_invariance'], median_invariance_rmse, validation['T5_invariance'])
        
        log_hypothesis("T5 Reduction", get_hypothesis_description('T5', 'T5_reduction'),
                      thresholds['T5_reduction'], entropy_reduction, validation['T5_reduction'])
        
        # Create plot
        fig = create_t5_plot(invariance_results, reduction_results, thresholds)
        
        # Prepare summary
        summary = {
            'test': 'T5',
            'parameters': {
                'sample_length': SAMPLE_LENGTH,
                'beta': BETA,
                'gamma_values': GAMMA_VALUES,
                'seeds': SEEDS,
                'usd_correlation': USD_STRENGTH_CORRELATION,
                'mapping_invariance': 'D2',
                'mapping_reduction': 'D1'
            },
            'results': {
                'invariance': {
                    'median_rmse': median_invariance_rmse,
                    'rmse_by_gamma': {gamma: np.median([r['joint_rmse'] for r in results.values()])
                                     for gamma, results in invariance_results.items()}
                },
                'pairwise_reduction': {
                    'target': reduction_results['target'],
                    'predictor': reduction_results['predictor'],
                    'entropy_reduction': entropy_reduction,
                    'marginal_entropy': reduction_results['marginal_entropy'],
                    'conditional_entropy': reduction_results['conditional_entropy']
                }
            },
            'hypotheses': {
                'T5_invariance': {
                    'pass': validation['T5_invariance'],
                    'metric': median_invariance_rmse,
                    'threshold': thresholds['T5_invariance'],
                    'description': get_hypothesis_description('T5', 'T5_invariance')
                },
                'T5_reduction': {
                    'pass': validation['T5_reduction'],
                    'metric': entropy_reduction,
                    'threshold': thresholds['T5_reduction'],
                    'description': get_hypothesis_description('T5', 'T5_reduction')
                }
            },
            'overall_pass': validation['T5_invariance'] and validation['T5_reduction']
        }
        
        # Prepare metrics for CSV - standardize all fields
        metrics = []
        for gamma, gamma_results in invariance_results.items():
            for pair, data in gamma_results.items():
                metrics.append({
                    'test_type': 'invariance',
                    'gamma': gamma,
                    'pair': pair,
                    'joint_rmse': data['joint_rmse'],
                    'target': '',
                    'predictor': '',
                    'entropy_reduction': 0.0,
                    'marginal_entropy': 0.0,
                    'conditional_entropy': 0.0,
                    'seed': SEEDS[0]
                })
        
        # Add reduction metrics with standardized fields
        metrics.append({
            'test_type': 'reduction',
            'gamma': 0.0,
            'pair': '',
            'joint_rmse': 0.0,
            'target': reduction_results['target'],
            'predictor': reduction_results['predictor'],
            'entropy_reduction': entropy_reduction,
            'marginal_entropy': reduction_results['marginal_entropy'],
            'conditional_entropy': reduction_results['conditional_entropy'],
            'seed': SEEDS[0]
        })
        
        # Save results
        summary_clean = format_json_safe(summary)
        save_test_results('T5', summary_clean, metrics, fig)
        
        # Log summary
        log_test_summary('T5', {
            'T5_invariance': validation['T5_invariance'],
            'T5_reduction': validation['T5_reduction']
        })
        
        return summary

def create_t5_plot(invariance_results: Dict, reduction_results: Dict, thresholds: Dict) -> plt.Figure:
    """Create T5 test visualization."""
    setup_plot_style()
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('T5: Market Slice Replication Test', fontsize=16, fontweight='bold')
    
    # Panel 1: Invariance RMSE by gamma
    ax1 = fig.add_subplot(gs[0, 0])
    
    gammas = list(invariance_results.keys())
    eurusd_rmses = [invariance_results[g]['EURUSD']['joint_rmse'] for g in gammas]
    gbpusd_rmses = [invariance_results[g]['GBPUSD']['joint_rmse'] for g in gammas]
    
    x = np.arange(len(gammas))
    width = 0.35
    
    ax1.bar(x - width/2, eurusd_rmses, width, label='EURUSD', alpha=0.7)
    ax1.bar(x + width/2, gbpusd_rmses, width, label='GBPUSD', alpha=0.7)
    ax1.axhline(y=thresholds.get('T5_invariance', 0.05), color='red', linestyle='--',
                label=f"Threshold: {thresholds.get('T5_invariance', 0.05):.3f}")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"γ={g}" for g in gammas])
    ax1.set_ylabel('Joint RMSE')
    ax1.set_title('Time-Scale Invariance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Sample triad evolution
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Show EURUSD triad for gamma=1.2
    if 1.2 in invariance_results:
        triads = invariance_results[1.2]['EURUSD']['triads_orig']
        time_subset = slice(0, min(500, len(triads['H'])))
        
        ax2.plot(triads['H'][time_subset], label='Entropy', alpha=0.7)
        ax2.plot(triads['C'][time_subset], label='Coherence', alpha=0.7)
        ax2.plot(triads['S'][time_subset], label='Stability', alpha=0.7)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Triad Value')
        ax2.set_title('EURUSD Triad Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Panel 3: Entropy reduction
    ax3 = fig.add_subplot(gs[0, 2])
    
    categories = ['Marginal\nH(EURUSD)', 'Conditional\nH(EURUSD|GBPUSD)']
    values = [reduction_results['marginal_entropy'], reduction_results['conditional_entropy']]
    
    bars = ax3.bar(categories, values, color=['blue', 'orange'], alpha=0.7)
    ax3.set_ylabel('Entropy (bits)')
    ax3.set_title('Pairwise Entropy Reduction')
    ax3.grid(True, alpha=0.3)
    
    # Add reduction text
    reduction = reduction_results['entropy_reduction']
    ax3.text(0.5, max(values) * 0.8, f'Reduction: {reduction:.1%}', 
             ha='center', transform=ax3.transData,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Panel 4: Market data sample
    ax4 = fig.add_subplot(gs[1, :2])
    
    # This would need actual FX data - show simulated data
    time_range = slice(0, min(1000, SAMPLE_LENGTH))
    ax4.plot(invariance_results[list(invariance_results.keys())[0]]['EURUSD']['triads_orig']['H'][time_range], 
             label='EURUSD Entropy', alpha=0.8)
    ax4.plot(invariance_results[list(invariance_results.keys())[0]]['GBPUSD']['triads_orig']['H'][time_range], 
             label='GBPUSD Entropy', alpha=0.8)
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Entropy')
    ax4.set_title('FX Entropy Time Series')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    invariance_pass = np.median([r['EURUSD']['joint_rmse'] for r in invariance_results.values()]) <= thresholds.get('T5_invariance', 0.05)
    reduction_pass = reduction_results['entropy_reduction'] >= thresholds.get('T5_reduction', 0.10)
    
    summary_text = "T5 Market Replication Results\n\n"
    summary_text += f"Invariance: {'PASS ✓' if invariance_pass else 'FAIL ✗'}\n"
    summary_text += f"Reduction: {'PASS ✓' if reduction_pass else 'FAIL ✗'}\n\n"
    summary_text += "Tests same invariances and\npairwise behaviors on synthetic\nFX data with USD common driver\n\n"
    summary_text += f"EURUSD-GBPUSD correlation\nvia USD strength: {USD_STRENGTH_CORRELATION:.1f}"
    
    ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    return fig

def main():
    """Main entry point."""
    result = run_t5_test()
    return result['overall_pass']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)