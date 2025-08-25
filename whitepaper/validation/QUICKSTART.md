# SEP Validation Framework - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
cd validation
pip install -r requirements.txt
```

### 2. Run Complete Validation
```bash
python run_sep_validation.py
```

### 3. Check Results
```bash
# View overall summary
cat results/latest/validation_summary.json

# Check individual test (example: T1)
ls results/latest/T1_time_scaling_test/
```

## Example Usage

### Basic Triad Computation
```python
#!/usr/bin/env python3
from sep_core import *
import numpy as np

# Generate a test signal (Van der Pol oscillator)
signal = generate_van_der_pol(length=10000, mu=1.0, seed=42)

# Convert to binary representation
bits = bit_mapping_D1(signal)

# Compute SEP triad observables (H, C, S)
triads = triad_series(bits, beta=0.1)

# Extract components
entropy = triads[:, 0]      # H_t - Information entropy
coherence = triads[:, 1]    # C_t - Pattern coherence  
stability = triads[:, 2]    # S_t - Temporal stability

print(f"Mean Entropy: {np.mean(entropy):.4f}")
print(f"Mean Coherence: {np.mean(coherence):.4f}")
print(f"Mean Stability: {np.mean(stability):.4f}")
```

### Running Individual Tests
```python
#!/usr/bin/env python3
# Example: Run just the time scaling test
from test_scripts import T1_time_scaling_test

# This will run T1 and show results
success = T1_time_scaling_test.main()
print(f"T1 Test Result: {'PASS' if success else 'FAIL'}")
```

### Custom Process Analysis
```python
#!/usr/bin/env python3
from sep_core import *
import numpy as np

# Use your own signal data
my_signal = np.loadtxt('my_data.txt')  # Replace with your data

# Test different bit mapping strategies
for strategy in ['D1', 'D2', 'D3']:
    if strategy == 'D1':
        bits = bit_mapping_D1(my_signal)
    elif strategy == 'D2':
        bits = bit_mapping_D2(my_signal)
    else:
        bits = bit_mapping_D3(my_signal)
    
    triads = triad_series(bits, beta=0.1)
    
    print(f"Strategy {strategy}:")
    print(f"  H mean: {np.mean(triads[:, 0]):.4f}")
    print(f"  C mean: {np.mean(triads[:, 1]):.4f}")
    print(f"  S mean: {np.mean(triads[:, 2]):.4f}")
```

## Expected Results

### Successful Validation Output
```
SEP PHYSICS VALIDATION FINAL REPORT
================================================================================
Timestamp: 2024-01-15T14:30:00.000000+00:00
Duration: 45.67 seconds

TEST RESULTS:
  Total Tests: 5
  Passed: 4
  Failed: 1
  Success Rate: 80.0%

HYPOTHESIS VALIDATION:
  Total Hypotheses: 10
  Passed: 8
  Failed: 2
  Success Rate: 80.0%

🎉 VALIDATION SUCCESSFUL: SEP physics claims supported by evidence
================================================================================
```

### What Each Test Does

| Test | Runtime | What It Validates |
|------|---------|-------------------|
| **T1** | ~8s | Time scaling properties of isolated vs reactive processes |
| **T2** | ~12s | Maximum entropy characteristics of pairwise triad distributions |
| **T3** | ~15s | Invariance under signal processing (convolution, decimation) |
| **T4** | ~10s | Reconstruction accuracy using triad-informed priors |
| **T5** | ~8s | Smoothing performance compared to naive filtering |

## Interpreting Results

### Files Generated
```
results/latest/
├── validation_summary.json     # Overall test status and timing
├── T1_time_scaling_test/
│   ├── T1_summary.json        # Detailed T1 results
│   ├── T1_time_scaling_metrics.csv    # Raw data for analysis
│   └── T1_plots.png          # Visualization of key findings
└── ... (similar for T2-T5)
```

### Key Metrics

**For Time Scaling (T1):**
- RMSE < 0.05 indicates good invariance
- Isolated processes should show stable triads
- Reactive processes should show scaling sensitivity

**For Maximum Entropy (T2):**
- Entropy gain > 10% indicates pairwise sufficiency
- Higher-order terms should provide minimal improvement

**For Convolution (T3):**
- Stability preservation > 80% indicates robust processing
- Band-limited signals should maintain triad structure

**For Reconstruction (T4):**
- Improvement > 20% over naive interpolation
- Gap RMSE should be significantly reduced

**For Smoothing (T5):**
- Uncertainty reduction > 15% over naive methods
- Parameter-stability correlation > 0.3

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're in the validation directory
cd validation
python -c "import sep_core; print('✓ sep_core imported successfully')"
```

**Missing Dependencies:**
```bash
pip install --upgrade numpy scipy matplotlib scikit-learn
```

**Permission Errors:**
```bash
# Make sure results directory is writable
mkdir -p results
chmod 755 results
```

**Memory Issues:**
If tests run out of memory, reduce process lengths in test files:
```python
# In test files, change:
PROCESS_LENGTH = 50000  # to smaller value like 10000
```

### Getting Help

1. **Check console output** - tests provide detailed progress information
2. **Examine JSON files** - contain complete parameter sets and results
3. **Review plots** - visual verification of test behavior
4. **Validate dependencies** - ensure all packages are correct versions

### Test-Specific Debugging

**T1 (Time Scaling):**
```python
# Quick validation of time scaling behavior
from sep_core import *
signal = generate_van_der_pol(1000)
bits = bit_mapping_D1(signal)
triads = triad_series(bits, beta=0.1)
print(f"Triad shape: {triads.shape}")  # Should be (998, 3)
```

**T2 (Maximum Entropy):**
```python
# Test entropy calculation
from sep_core import compute_joint_entropy
entropy = compute_joint_entropy([0,1,0,1,1,0], [1,0,1,0,0,1])
print(f"Joint entropy: {entropy}")  # Should be reasonable value
```

## Next Steps

### For Research Use:
1. Run validation on your specific data domains
2. Modify parameters to match your signal characteristics
3. Extend tests for domain-specific hypotheses
4. Use results to inform SEP-based algorithm development

### For Development:
1. Add new test cases following the existing pattern
2. Implement alternative triad metrics
3. Explore different bit mapping strategies
4. Integrate with existing signal processing pipelines

### For Publication:
1. Document any parameter changes made
2. Report full validation results
3. Include computational environment details
4. Provide data availability statements