"""
Plotting utilities for SEP validation tests.
Provides standardized multi-panel plots and paper-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import seaborn as sns

# Set consistent style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Standard figure parameters
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 8
DPI = 150
FONT_SIZE = 10

def setup_plot_style():
    """Setup consistent plotting style."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE + 2,
        'axes.titlesize': FONT_SIZE + 4,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'figure.titlesize': FONT_SIZE + 6,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'grid.alpha': 0.3,
        'axes.grid': True
    })

def plot_triad_evolution(triads: Dict[str, np.ndarray], 
                        title: str = "Triad Evolution",
                        time_axis: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Plot H, C, S evolution over time.
    
    Args:
        triads: Dictionary with keys 'H', 'C', 'S'
        title: Figure title
        time_axis: Optional time array for x-axis
        
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if time_axis is None:
        time_axis = np.arange(len(triads['H']))
    
    # Plot entropy
    axes[0].plot(time_axis, triads['H'], 'b-', label='Entropy (H)', alpha=0.8)
    axes[0].set_ylabel('Entropy (bits)')
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot coherence
    axes[1].plot(time_axis, triads['C'], 'g-', label='Coherence (C)', alpha=0.8)
    axes[1].set_ylabel('Coherence')
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot stability
    axes[2].plot(time_axis, triads['S'], 'r-', label='Stability (S)', alpha=0.8)
    axes[2].set_ylabel('Stability')
    axes[2].set_xlabel('Time')
    axes[2].set_ylim([0, 1])
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_t1_results(results: Dict, thresholds: Dict) -> plt.Figure:
    """
    Create standard T1 test figure with isolation vs reactive comparison.
    
    Args:
        results: Dictionary with test results
        thresholds: Dictionary with pass/fail thresholds
        
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(2, 2, figure=fig)
    
    # Title
    fig.suptitle('T1: Time-Scaling Invariance Test', fontsize=14, fontweight='bold')
    
    # Panel 1: Isolated RMSE vs gamma
    ax1 = fig.add_subplot(gs[0, 0])
    gammas = results.get('gammas', [1.2, 1.5, 2.0])
    isolated_rmse = results.get('isolated_rmse', [])
    
    ax1.bar(range(len(gammas)), isolated_rmse, color='blue', alpha=0.7)
    ax1.axhline(y=thresholds.get('H1', 0.05), color='red', linestyle='--', 
                label=f"Threshold: {thresholds.get('H1', 0.05):.3f}")
    ax1.set_xticks(range(len(gammas)))
    ax1.set_xticklabels([f"γ={g}" for g in gammas])
    ax1.set_ylabel('Joint RMSE')
    ax1.set_title('Isolated Process (H1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Reactive RMSE vs gamma
    ax2 = fig.add_subplot(gs[0, 1])
    reactive_rmse = results.get('reactive_rmse', [])
    
    ax2.bar(range(len(gammas)), reactive_rmse, color='orange', alpha=0.7)
    ax2.set_xticks(range(len(gammas)))
    ax2.set_xticklabels([f"γ={g}" for g in gammas])
    ax2.set_ylabel('Joint RMSE')
    ax2.set_title('Reactive Process')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: RMSE Ratio
    ax3 = fig.add_subplot(gs[1, 0])
    ratios = results.get('rmse_ratios', [])
    
    ax3.bar(range(len(gammas)), ratios, color='green', alpha=0.7)
    ax3.axhline(y=thresholds.get('H2', 2.0), color='red', linestyle='--',
                label=f"Threshold: {thresholds.get('H2', 2.0):.1f}")
    ax3.set_xticks(range(len(gammas)))
    ax3.set_xticklabels([f"γ={g}" for g in gammas])
    ax3.set_ylabel('Reactive/Isolated Ratio')
    ax3.set_title('Reactive vs Isolated (H2)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary status
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    h1_pass = results.get('H1_pass', False)
    h2_pass = results.get('H2_pass', False)
    
    status_text = f"Test Results:\n\n"
    status_text += f"H1 (Isolation): {'PASS ✓' if h1_pass else 'FAIL ✗'}\n"
    status_text += f"  Median RMSE: {results.get('isolated_median', 0):.4f}\n\n"
    status_text += f"H2 (Reactive Break): {'PASS ✓' if h2_pass else 'FAIL ✗'}\n"
    status_text += f"  Median Ratio: {results.get('reactive_ratio', 0):.2f}\n\n"
    status_text += f"Overall: {'PASS' if h1_pass and h2_pass else 'FAIL'}"
    
    ax4.text(0.1, 0.5, status_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    return fig

def plot_t2_results(results: Dict, thresholds: Dict) -> plt.Figure:
    """
    Create standard T2 test figure with pairwise sufficiency analysis.
    
    Args:
        results: Dictionary with test results
        thresholds: Dictionary with pass/fail thresholds
        
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(2, 2, figure=fig)
    
    # Title
    fig.suptitle('T2: Pairwise Maximum-Entropy Sufficiency Test', 
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Reduction vs rho curve
    ax1 = fig.add_subplot(gs[0, :])
    rhos = results.get('rhos', [0.0, 0.2, 0.4, 0.6, 0.8])
    reductions = results.get('reductions', [])
    
    ax1.plot(rhos, reductions, 'bo-', markersize=8, linewidth=2, label='Observed')
    ax1.axhline(y=thresholds.get('H3', 0.30), color='red', linestyle='--',
                label=f"H3 Threshold: {thresholds.get('H3', 0.30):.2f}")
    ax1.fill_between(rhos, 0, reductions, alpha=0.3)
    ax1.set_xlabel('Correlation (ρ)')
    ax1.set_ylabel('Relative Entropy Reduction')
    ax1.set_title('Information Captured by Pairwise Conditioning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(0.5, max(reductions) * 1.2) if reductions else 0.5])
    
    # Panel 2: Order-2 excess histogram
    ax2 = fig.add_subplot(gs[1, 0])
    excess = results.get('order2_excess', [])
    
    if excess:
        ax2.hist(excess, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax2.axvline(x=thresholds.get('H4', 0.05), color='red', linestyle='--',
                   label=f"H4 Threshold: {thresholds.get('H4', 0.05):.3f}")
        ax2.set_xlabel('Normalized Order-2 Excess')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Higher-Order Contribution (H4)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Panel 3: Summary status
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    h3_pass = results.get('H3_pass', False)
    h4_pass = results.get('H4_pass', False)
    
    status_text = f"Test Results:\n\n"
    status_text += f"H3 (Pairwise Reduction): {'PASS ✓' if h3_pass else 'FAIL ✗'}\n"
    status_text += f"  Max Reduction: {results.get('max_reduction', 0):.3f}\n\n"
    status_text += f"H4 (Order-2 Excess): {'PASS ✓' if h4_pass else 'FAIL ✗'}\n"
    status_text += f"  Median Excess: {results.get('median_excess', 0):.4f}\n\n"
    status_text += f"Overall: {'PASS' if h3_pass and h4_pass else 'FAIL'}"
    
    ax3.text(0.1, 0.5, status_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    return fig

def plot_t3_results(results: Dict, thresholds: Dict) -> plt.Figure:
    """
    Create standard T3 test figure for convolution invariance.
    
    Args:
        results: Dictionary with test results
        thresholds: Dictionary with pass/fail thresholds
        
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(2, 2, figure=fig)
    
    # Title
    fig.suptitle('T3: Convolution Invariance Test', fontsize=14, fontweight='bold')
    
    # Panel 1: With antialiasing
    ax1 = fig.add_subplot(gs[0, 0])
    decimations = results.get('decimations', [2, 4])
    rmse_aa = results.get('rmse_antialiased', [])
    
    ax1.bar(range(len(decimations)), rmse_aa, color='green', alpha=0.7)
    ax1.axhline(y=thresholds.get('T3', 0.05), color='red', linestyle='--',
                label=f"Threshold: {thresholds.get('T3', 0.05):.3f}")
    ax1.set_xticks(range(len(decimations)))
    ax1.set_xticklabels([f"÷{d}" for d in decimations])
    ax1.set_ylabel('Joint RMSE')
    ax1.set_title('With Antialiasing (PASS)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Without antialiasing (control)
    ax2 = fig.add_subplot(gs[0, 1])
    rmse_no_aa = results.get('rmse_no_antialiasing', [])
    
    ax2.bar(range(len(decimations)), rmse_no_aa, color='red', alpha=0.7)
    ax2.axhline(y=thresholds.get('T3', 0.05), color='red', linestyle='--',
                label=f"Threshold: {thresholds.get('T3', 0.05):.3f}")
    ax2.set_xticks(range(len(decimations)))
    ax2.set_xticklabels([f"÷{d}" for d in decimations])
    ax2.set_ylabel('Joint RMSE')
    ax2.set_title('Without Antialiasing (FAIL)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Signal comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    if 'original_triad' in results and 'decimated_triad' in results:
        time_orig = np.arange(len(results['original_triad']['H']))
        time_dec = np.arange(len(results['decimated_triad']['H'])) * 2
        
        ax3.plot(time_orig[:200], results['original_triad']['H'][:200], 
                'b-', alpha=0.5, label='Original')
        ax3.plot(time_dec[:100], results['decimated_triad']['H'][:100], 
                'r--', alpha=0.8, label='Decimated ÷2')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Entropy')
        ax3.set_title('Triad Preservation Example')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_comparison(triads1: Dict[str, np.ndarray], 
                   triads2: Dict[str, np.ndarray],
                   label1: str = "Original",
                   label2: str = "Transformed",
                   title: str = "Triad Comparison") -> plt.Figure:
    """
    Plot side-by-side comparison of two triads.
    
    Args:
        triads1: First triad dictionary
        triads2: Second triad dictionary
        label1: Label for first triad
        label2: Label for second triad
        title: Figure title
        
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Ensure same length for comparison
    min_len = min(len(triads1['H']), len(triads2['H']))
    
    # Plot entropy comparison
    axes[0].plot(triads1['H'][:min_len], 'b-', alpha=0.6, label=f'{label1} H')
    axes[0].plot(triads2['H'][:min_len], 'r--', alpha=0.8, label=f'{label2} H')
    axes[0].set_ylabel('Entropy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot coherence comparison
    axes[1].plot(triads1['C'][:min_len], 'b-', alpha=0.6, label=f'{label1} C')
    axes[1].plot(triads2['C'][:min_len], 'r--', alpha=0.8, label=f'{label2} C')
    axes[1].set_ylabel('Coherence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot stability comparison
    axes[2].plot(triads1['S'][:min_len], 'b-', alpha=0.6, label=f'{label1} S')
    axes[2].plot(triads2['S'][:min_len], 'r--', alpha=0.8, label=f'{label2} S')
    axes[2].set_ylabel('Stability')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_paper_figure(test_name: str, results: Dict, thresholds: Dict) -> plt.Figure:
    """
    Create paper-ready figure for a given test.
    
    Args:
        test_name: Name of the test (T1, T2, etc.)
        results: Test results dictionary
        thresholds: Thresholds dictionary
        
    Returns:
        Figure object
    """
    if test_name == 'T1':
        return plot_t1_results(results, thresholds)
    elif test_name == 'T2':
        return plot_t2_results(results, thresholds)
    elif test_name == 'T3':
        return plot_t3_results(results, thresholds)
    else:
        # Default to triad evolution plot
        return plot_triad_evolution(results.get('triads', {}), 
                                   title=f"{test_name} Results")

def export_paper_figures(results_dir: str = 'whitepaper/validation/results',
                        output_dir: str = 'docs/figures'):
    """
    Export all paper figures from test results.
    
    Args:
        results_dir: Directory containing test results
        output_dir: Directory to save paper figures
    """
    from pathlib import Path
    import json
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    tests = ['T1', 'T2', 'T3', 'T4', 'T5']
    
    for test in tests:
        summary_file = Path(results_dir) / test / f"{test}_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                results = json.load(f)
            
            # Load thresholds (you'll need to import from thresholds.py)
            from validation import thresholds
            test_thresholds = thresholds.get_thresholds(test)
            
            # Create figure
            fig = create_paper_figure(test, results, test_thresholds)
            
            # Save figure
            output_path = Path(output_dir) / f"{test}_paper_figure.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Exported {test} figure to {output_path}")