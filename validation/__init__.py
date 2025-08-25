"""
SEP Physics Validation Framework

This package provides a comprehensive validation suite for the SEP physics theory,
including tests for time-scaling invariance, pairwise sufficiency, and convolution invariance.
"""

__version__ = "1.0.0"
__author__ = "SEP Research Team"

# Import key components for easy access
from .common import (
    compute_triad,
    mapping_D1_derivative_sign,
    mapping_D2_dilation_robust,
    gaussian_entropy_bits,
    compute_joint_rmse
)

from .thresholds import (
    get_thresholds,
    check_hypothesis,
    get_all_test_names
)

__all__ = [
    'compute_triad',
    'mapping_D1_derivative_sign', 
    'mapping_D2_dilation_robust',
    'gaussian_entropy_bits',
    'compute_joint_rmse',
    'get_thresholds',
    'check_hypothesis',
    'get_all_test_names'
]