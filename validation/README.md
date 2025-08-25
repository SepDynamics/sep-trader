# SEP Physics Validation Plan - Python Implementation

## Overview

This directory contains a **comprehensive Python-only implementation** of the **SEP (Signal Enhancement Physics) Validation Plan**. The implementation has transitioned from the original C++ framework due to build impediments encountered during CUDA compiler identification, adopting a **Python-centric validation workflow** that provides **publishable-grade, deterministic, and reproducible validation protocols**.

## Architecture

### Core Components

#### 1. **SEP Core Module** (`sep_core.py`)
The foundational module implementing:
- **Triad Observables**: Coherence (C_t), Stability (S_t), and Entropy (H_t) with Exponential Moving Average (EMA)
- **Bit Mapping Strategies**: D1, D2, D3 for signal-to-binary transduction
- **Process Generation**: Synthetic, physically motivated processes (Poisson, Van der Pol, Chirp)
- **Utility Functions**: RMSE calculation, controlled convolution via decimation

#### 2. **Test Suite** (T1-T5)
Five comprehensive test modules validating specific SEP hypotheses:

- **T1**: [`test_T1_time_scaling.py`](validation/test_T1_time_scaling.py) - Isolated vs Reactive Time Scaling
- **T2**: [`test_T2_maxent_sufficiency.py`](validation/test_T2_maxent_sufficiency.py) - Pairwise Maximum-Entropy Sufficiency  
- **T3**: [`test_T3_convolutional_invariance.py`](validation/test_T3_convolutional_invariance.py) - Convolutional Invariance on Band-Limited Waves
- **T4**: [`test_T4_retrodictive_reconstruction.py`](validation/test_T4_retrodictive_reconstruction.py) - Retrodictive Reconstruction With Continuity Constraint
- **T5**: [`test_T5_smoothing_beats_filtering.py`](validation/test_T5_smoothing_beats_filtering.py) - Smoothing Beats Filtering (uncertainty reduction)

#### 3. **Validation Framework** (`run_sep_validation.py`)
Automated test runner with:
- Dependency checking
- Timestamped results generation
- Comprehensive reporting
- Artifact management

## Hypotheses Being Tested

| Test | Hypothesis | Description |
|------|------------|-------------|
| T1 | **H1** | Isolated processes show invariant triad trajectories under time scaling |
| T1 | **H2** | Reactive processes break triad alignment under naive time scaling |
| T2 | **H3** | Pairwise joint triads achieve maximum entropy distribution |
| T2 | **H4** | Adding more variables does not significantly improve entropy beyond pairwise |
| T3 | **H5** | Triad metrics remain stable under controlled decimation of band-limited waves |
| T3 | **H6** | Convolutional operations preserve triad structure for appropriately band-limited signals |
| T4 | **H7** | Retrodictive reconstruction using triad-informed priors outperforms naive interpolation |
| T4 | **H8** | Continuity constraints improve reconstruction accuracy for smooth underlying processes |
| T5 | **H9** | SEP-informed smoothing outperforms naive filtering in uncertainty reduction |
| T5 | **H10** | Optimal smoothing parameters correlate with triad stability metrics |

## Installation & Setup

### 1. Dependencies
Install required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy >= 1.21.0`
- `scipy >= 1.7.0` 
- `scikit-learn >= 1.0.0`
- `matplotlib >= 3.5.0`

### 2. Verification
Check installation:

```bash
python run_sep_validation.py
```

The framework will automatically check dependencies and provide feedback.

## Usage

### Running Individual Tests

Each test can be run independently:

```bash
# Test T1: Time Scaling
python test_T1_time_scaling.py

# Test T2: Maximum Entropy
python test_T2_maxent_sufficiency.py

# Test T3: Convolutional Invariance  
python test_T3_convolutional_invariance.py

# Test T4: Retrodictive Reconstruction
python test_T4_retrodictive_reconstruction.py

# Test T5: Smoothing vs Filtering
python test_T5_smoothing_beats_filtering.py
```

Each test generates:
- **JSON summary** with detailed results
- **CSV metrics** for quantitative analysis
- **Visualization plots** showing key findings
- **Console output** with pass/fail status

### Running Complete Validation Suite

For comprehensive validation:

```bash
python run_sep_validation.py
```

This will:
1. Check all dependencies
2. Create timestamped results directory
3. Run all tests sequentially
4. Generate comprehensive summary report
5. Provide overall validation status

### Results Structure

```
results/
├── validation_run_YYYYMMDD_HHMMSS_UTC/
│   ├── validation_summary.json          # Overall summary
│   ├── test_T1_time_scaling/
│   │   ├── T1_summary.json              # Test-specific results
│   │   ├── T1_time_scaling_metrics.csv  # Quantitative data
│   │   └── T1_plots.png                 # Visualizations
│   ├── test_T2_maxent_sufficiency/
│   │   ├── T2_summary.json
│   │   ├── T2_entropy_analysis.csv
│   │   └── T2_plots.png
│   └── ... (similar for T3, T4, T5)
└── latest -> validation_run_YYYYMMDD_HHMMSS_UTC/
```

## Core Functions

### Triad Computation

```python
from sep_core import triad_series, bit_mapping_D1

# Generate signal
signal = generate_van_der_pol(length=10000, mu=1.0)

# Convert to binary
bits = bit_mapping_D1(signal)

# Compute triad series (H, C, S)
triads = triad_series(bits, beta=0.1)

# Extract components
entropy = triads[:, 0]      # H_t
coherence = triads[:, 1]    # C_t  
stability = triads[:, 2]    # S_t
```

### Process Generation

```python
from sep_core import generate_poisson_process, generate_van_der_pol, generate_chirp

# Poisson process (isolated)
poisson = generate_poisson_process(length=50000, rate=1.0, seed=42)

# Van der Pol oscillator (reactive)
vdp = generate_van_der_pol(length=50000, mu=1.0, seed=42)

# Chirp signal (controlled frequency sweep)
chirp = generate_chirp(length=50000, f0=0.005, f1=0.02, seed=42)
```

### Bit Mapping Strategies

```python
from sep_core import bit_mapping_D1, bit_mapping_D2, bit_mapping_D3

signal = np.random.randn(1000)

# D1: Simple threshold at median
bits_d1 = bit_mapping_D1(signal)

# D2: Local gradient-based mapping
bits_d2 = bit_mapping_D2(signal) 

# D3: Multi-scale threshold mapping
bits_d3 = bit_mapping_D3(signal)
```

## Validation Metrics

### Test-Specific Metrics

- **RMSE**: Root Mean Square Error for signal reconstruction quality
- **Time Scaling Invariance**: Deviation under temporal transformations
- **Entropy Gain**: Information-theoretic improvements
- **Convolutional Stability**: Preservation under signal processing
- **Reconstruction Accuracy**: Gap-filling performance
- **Uncertainty Reduction**: Smoothing effectiveness
- **Correlation Coefficients**: Parameter-stability relationships

### Success Criteria

Each hypothesis has specific **quantitative thresholds**:

- **H1/H2**: RMSE deviation < 0.05 for time scaling invariance
- **H3/H4**: Entropy gain > 10% for pairwise sufficiency
- **H5/H6**: Stability preservation > 80% under convolution
- **H7/H8**: Reconstruction improvement > 20% over baseline
- **H9/H10**: Uncertainty reduction > 15%, correlation > 0.3

## Reproducibility

### Deterministic Execution
- **Fixed random seeds** for all stochastic processes
- **Versioned dependencies** in [`requirements.txt`](validation/requirements.txt)
- **Timestamped results** with complete parameter logging
- **Platform-independent** Python implementation

### Artifact Generation
- **JSON data** for programmatic analysis
- **CSV exports** for statistical software
- **High-resolution plots** for publication
- **Comprehensive logs** with timing information

### Result Verification
```bash
# Check specific test results
cat results/latest/test_T1_time_scaling/T1_summary.json

# View validation summary  
cat results/latest/validation_summary.json

# Analyze metrics in spreadsheet software
open results/latest/test_T1_time_scaling/T1_time_scaling_metrics.csv
```

## Scientific Methodology

### Falsifiable Testing
Each hypothesis is structured as a **falsifiable claim** with:
- **Null hypothesis** (H0): SEP effects are not present
- **Alternative hypothesis** (H1): SEP effects exceed threshold
- **Statistical significance** testing where applicable
- **Effect size** quantification

### Pre-registered Analysis
- **Hypothesis formulation** precedes implementation
- **Success criteria** defined before testing
- **Analysis pipeline** automated to prevent bias
- **Complete methodology** documented in code

### Peer Review Compatibility
Results formatted for:
- **Academic publication** standards
- **Open science** practices
- **Replication** by independent researchers
- **Extension** to new domains

## Interpretation

### Positive Results
If validation succeeds (≥80% tests pass, ≥70% hypotheses supported):
- **SEP principles** show empirical support
- **Triad metrics** demonstrate predictive power
- **Signal processing** applications feasible
- **Further research** warranted

### Negative Results  
If validation fails:
- **Hypotheses require refinement** or rejection
- **Implementation issues** need investigation
- **Alternative approaches** should be explored
- **Null results** are scientifically valuable

### Mixed Results
Partial success indicates:
- **Domain-specific** applicability
- **Parameter sensitivity** requiring tuning  
- **Scale-dependent** phenomena
- **Conditional validity** of claims

## Extensions

### Adding New Tests
1. Create test module following naming pattern `test_T*_description.py`
2. Implement [`main()`](validation/run_sep_validation.py:main) function returning boolean success
3. Generate results in standardized format (JSON + CSV + plots)
4. Add to test suite in [`run_sep_validation.py`](validation/run_sep_validation.py)

### Custom Processes
```python
def generate_custom_process(length, params, seed=42):
    """Generate custom process for SEP testing."""
    np.random.seed(seed)
    # Your implementation here
    return signal_array
```

### Alternative Metrics
```python
def custom_triad_metric(bits, window_size=100):
    """Compute alternative to H/C/S triad."""
    # Your metric computation
    return metric_series
```

## Citation

If this validation framework contributes to your research, please cite:

```
SEP Physics Validation Plan - Python Implementation
[Include appropriate academic citation when published]
```

## Support

For questions or issues:
1. Check test outputs in `results/latest/`
2. Review console error messages
3. Verify dependency versions
4. Examine individual test modules for implementation details

## License

[Include appropriate license information]

---

**Note**: This implementation represents a **strategic pivot** from C++ to Python, maintaining **scientific rigor** while ensuring **accessibility** and **reproducibility** for the **SEP Physics Validation Plan**.