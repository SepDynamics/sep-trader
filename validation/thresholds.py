"""
Thresholds and pass/fail criteria for SEP validation tests.
Single source of truth for all test thresholds.
"""

from typing import Dict, Any

# T1: Time-Scaling Invariance Test
T1_THRESHOLDS = {
    'H1': 0.05,  # Max RMSE for isolated processes
    'H2': 2.0,   # Min ratio of reactive/isolated RMSE
    'description': {
        'H1': 'Isolated processes should maintain triad invariance (RMSE ≤ 0.05)',
        'H2': 'Reactive processes should break invariance (ratio ≥ 2.0)'
    }
}

# T2: Pairwise Maximum-Entropy Sufficiency Test
T2_THRESHOLDS = {
    'H3': 0.30,  # Min relative entropy reduction for coupled processes
    'H4': 0.05,  # Max normalized order-2 excess
    'rho_threshold': 0.6,  # Correlation threshold for H3
    'description': {
        'H3': 'Pairwise conditioning should capture ≥30% information at ρ≥0.6',
        'H4': 'Higher-order terms should contribute ≤5% additional information'
    }
}

# T3: Convolution Invariance Test
T3_THRESHOLDS = {
    'T3': 0.05,  # Max RMSE after antialiased decimation
    'decimation_factors': [2, 4],  # Decimation factors to test
    'description': {
        'T3': 'Antialiased decimation should preserve triad (RMSE ≤ 0.05)'
    }
}

# T4: Retrodiction Uniqueness Test
T4_THRESHOLDS = {
    'T4_uniqueness': 0.95,  # Min uniqueness rate
    'flip_budget': 2,  # Max Hamming distance k
    'window_length': 16,  # Min window length L
    'description': {
        'T4': 'Should achieve ≥95% uniqueness for k≤2, L≥16'
    }
}

# T5: Market Slice Replication Test
T5_THRESHOLDS = {
    'T5_invariance': 0.05,  # Max RMSE for time-scaled market data
    'T5_reduction': 0.10,   # Min entropy reduction for FX pairs
    'description': {
        'T5_invariance': 'Market data should show similar invariance (RMSE ≤ 0.05)',
        'T5_reduction': 'FX pairs should show ≥10% reduction (USD common driver)'
    }
}

# Consolidated thresholds
ALL_THRESHOLDS = {
    'T1': T1_THRESHOLDS,
    'T2': T2_THRESHOLDS,
    'T3': T3_THRESHOLDS,
    'T4': T4_THRESHOLDS,
    'T5': T5_THRESHOLDS
}

def get_thresholds(test_name: str) -> Dict[str, Any]:
    """
    Get thresholds for a specific test.
    
    Args:
        test_name: Name of the test (T1, T2, etc.)
        
    Returns:
        Dictionary of thresholds for the test
    """
    return ALL_THRESHOLDS.get(test_name, {})

def check_hypothesis(test_name: str, hypothesis: str, metric: float) -> bool:
    """
    Check if a hypothesis passes based on the metric.
    
    Args:
        test_name: Name of the test
        hypothesis: Hypothesis name (H1, H2, etc.)
        metric: Computed metric value
        
    Returns:
        True if hypothesis passes, False otherwise
    """
    thresholds = get_thresholds(test_name)
    
    if hypothesis not in thresholds:
        return False
    
    threshold = thresholds[hypothesis]
    
    # Determine comparison based on hypothesis
    if hypothesis in ['H1', 'H4', 'T3', 'T5_invariance']:
        # These require metric to be less than or equal to threshold
        return metric <= threshold
    elif hypothesis in ['H2', 'H3', 'T4_uniqueness', 'T5_reduction']:
        # These require metric to be greater than or equal to threshold
        return metric >= threshold
    else:
        return False

def get_hypothesis_description(test_name: str, hypothesis: str) -> str:
    """
    Get description of a hypothesis.
    
    Args:
        test_name: Name of the test
        hypothesis: Hypothesis name
        
    Returns:
        Description string
    """
    thresholds = get_thresholds(test_name)
    descriptions = thresholds.get('description', {})
    return descriptions.get(hypothesis, f"{hypothesis} for {test_name}")

def format_threshold_summary(test_name: str) -> str:
    """
    Format a summary of all thresholds for a test.
    
    Args:
        test_name: Name of the test
        
    Returns:
        Formatted string summary
    """
    thresholds = get_thresholds(test_name)
    
    if not thresholds:
        return f"No thresholds defined for {test_name}"
    
    lines = [f"Thresholds for {test_name}:"]
    
    for key, value in thresholds.items():
        if key == 'description':
            continue
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                lines.append(f"  {subkey}: {subvalue}")
        else:
            lines.append(f"  {key}: {value}")
    
    # Add descriptions
    descriptions = thresholds.get('description', {})
    if descriptions:
        lines.append("\nDescriptions:")
        for hyp, desc in descriptions.items():
            lines.append(f"  {hyp}: {desc}")
    
    return "\n".join(lines)

# Test-specific validation functions
def validate_t1_results(isolated_rmse: float, reactive_rmse: float) -> Dict[str, bool]:
    """
    Validate T1 test results.
    
    Args:
        isolated_rmse: Median RMSE for isolated process
        reactive_rmse: Median RMSE for reactive process
        
    Returns:
        Dictionary with H1 and H2 pass/fail status
    """
    h1_pass = check_hypothesis('T1', 'H1', isolated_rmse)
    
    # Calculate ratio
    ratio = reactive_rmse / (isolated_rmse + 1e-10)
    h2_pass = check_hypothesis('T1', 'H2', ratio)
    
    return {
        'H1': h1_pass,
        'H2': h2_pass,
        'ratio': ratio
    }

def validate_t2_results(reductions: Dict[float, float], 
                       order2_excess: float) -> Dict[str, bool]:
    """
    Validate T2 test results.
    
    Args:
        reductions: Dictionary mapping rho to reduction values
        order2_excess: Median order-2 excess
        
    Returns:
        Dictionary with H3 and H4 pass/fail status
    """
    # Check H3: need reduction ≥ 0.30 at rho ≥ 0.6
    rho_threshold = T2_THRESHOLDS['rho_threshold']
    h3_threshold = T2_THRESHOLDS['H3']
    
    h3_pass = False
    max_reduction = 0
    
    for rho, reduction in reductions.items():
        if rho >= rho_threshold and reduction >= h3_threshold:
            h3_pass = True
        max_reduction = max(max_reduction, reduction)
    
    # Check H4
    h4_pass = check_hypothesis('T2', 'H4', order2_excess)
    
    return {
        'H3': h3_pass,
        'H4': h4_pass,
        'max_reduction': max_reduction
    }

def validate_t3_results(rmse_with_aa: float, rmse_without_aa: float) -> Dict[str, bool]:
    """
    Validate T3 test results.
    
    Args:
        rmse_with_aa: RMSE with antialiasing
        rmse_without_aa: RMSE without antialiasing
        
    Returns:
        Dictionary with T3 pass/fail status
    """
    t3_pass = check_hypothesis('T3', 'T3', rmse_with_aa)
    control_fail = rmse_without_aa > T3_THRESHOLDS['T3']
    
    return {
        'T3': t3_pass,
        'control_fail': control_fail,
        'rmse_ratio': rmse_without_aa / (rmse_with_aa + 1e-10)
    }

def validate_t4_results(uniqueness_rates: Dict[tuple, float]) -> Dict[str, bool]:
    """
    Validate T4 test results.
    
    Args:
        uniqueness_rates: Dictionary mapping (k, L) to uniqueness rate
        
    Returns:
        Dictionary with T4 pass/fail status
    """
    k_threshold = T4_THRESHOLDS['flip_budget']
    l_threshold = T4_THRESHOLDS['window_length']
    uniqueness_threshold = T4_THRESHOLDS['T4_uniqueness']
    
    t4_pass = False
    
    for (k, l), rate in uniqueness_rates.items():
        if k <= k_threshold and l >= l_threshold:
            if rate >= uniqueness_threshold:
                t4_pass = True
                break
    
    return {
        'T4': t4_pass,
        'best_rate': max(uniqueness_rates.values()) if uniqueness_rates else 0
    }

def validate_t5_results(invariance_rmse: float, entropy_reduction: float) -> Dict[str, bool]:
    """
    Validate T5 test results.
    
    Args:
        invariance_rmse: RMSE for time-scaled market data
        entropy_reduction: Entropy reduction for FX pairs
        
    Returns:
        Dictionary with T5 pass/fail status
    """
    t5_invariance_pass = check_hypothesis('T5', 'T5_invariance', invariance_rmse)
    t5_reduction_pass = check_hypothesis('T5', 'T5_reduction', entropy_reduction)
    
    return {
        'T5_invariance': t5_invariance_pass,
        'T5_reduction': t5_reduction_pass
    }

# Export convenience function
def get_all_test_names():
    """Get list of all test names."""
    return list(ALL_THRESHOLDS.keys())

def summary_table() -> str:
    """
    Generate a summary table of all thresholds.
    
    Returns:
        Formatted table string
    """
    lines = ["SEP Validation Thresholds Summary", "=" * 50]
    
    for test_name in get_all_test_names():
        lines.append(f"\n{test_name}:")
        thresholds = get_thresholds(test_name)
        
        for key, value in thresholds.items():
            if key == 'description':
                continue
            lines.append(f"  {key:20s}: {value}")
    
    return "\n".join(lines)