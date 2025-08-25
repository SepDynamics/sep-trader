# T4: Retrodictive Reconstruction Testing Report

## Overview

The T4 test evaluates whether triad-informed reconstruction can outperform naive interpolation methods and whether continuity constraints can further improve reconstruction quality. This test is crucial for validating the practical utility of the triad observables in signal reconstruction applications.

## Test Results Summary

* **H7 (Triad-informed outperforms naive)**: Median improvement ratio **-27.2%** (FAIL vs 20% target threshold).
* **H8 (Continuity constraints improve)**: Median improvement ratio **4.3%** (FAIL vs 15% target threshold).
* **Overall**: **FAIL**

## Detailed Results

### Hypothesis H7: Triad-informed reconstruction outperforms naive interpolation

The triad-informed reconstruction method consistently underperformed compared to simple linear interpolation:

* Median linear RMSE: 0.4070981397887319
* Median triad-informed RMSE: 7.5281670875635545
* Median improvement ratio: -27.20196749069272%
* Threshold: 0.2 (20% improvement required)
* Number of comparisons: 16
* Result: **FAIL**

### Hypothesis H8: Continuity constraints improve reconstruction

Adding continuity constraints provided minimal improvement:

* Median cubic RMSE: 4.0128684775071815
* Median constrained RMSE: 3.8639412791140533
* Median improvement ratio: 0.043172650881608224 (4.3%)
* Threshold: 0.15 (15% improvement required)
* Number of comparisons: 16
* Result: **FAIL**

## Data and Methodology

### Test Data

The test was conducted on multiple signal types:
* Poisson (random) processes
* van der Pol (reactive) processes
* Chirp (structured) signals

### Mappings

Both bit mappings were tested:
* D1 (derivative sign)
* D2 (rolling quantiles)

### Gap Configurations

Various gap sizes and positions were tested:
* Gap sizes: 100, 200, 500, and 1000 samples
* Gap positions: beginning, middle, and end of the signal

### Reconstruction Methods

Four reconstruction methods were compared:
1. **Linear**: Simple linear interpolation
2. **Cubic**: Cubic spline interpolation
3. **Triad-informed**: Reconstruction using triad observable patterns
4. **Constrained**: Triad-informed reconstruction with continuity constraints

## Key Issues Identified

### 1. Triad-informed reconstruction underperformance

The triad-informed method consistently underperformed compared to simple linear interpolation, which is unexpected and suggests fundamental issues with the current approach.

### 2. Extreme values in some cases

Some reconstruction methods (particularly cubic and constrained) produced extremely high RMSE values in certain configurations, indicating numerical instability or failure cases.

### 3. Mapping sensitivity

Results varied significantly between D1 and D2 mappings, with D2 generally producing more stable but less accurate results.

## Scientific Value

While the T4 test did not pass its hypotheses, it provides valuable diagnostic information about the limitations of the current triad-informed reconstruction approach:

1. The consistent failure of H7 indicates that the triad observables may not be directly applicable to reconstruction in the way currently implemented, or that the reconstruction algorithm needs significant refinement.

2. The failure of H8 suggests that the continuity constraints are not effectively capturing the underlying signal structure.

3. The test revealed numerical stability issues that need to be addressed in future implementations.

## Next Steps

Based on the T4 results, the following actions are recommended:

1. **Investigate triad-informed reconstruction approach**: Analyze why the triad observables are not effectively leveraged for reconstruction and explore alternative algorithms.

2. **Address numerical stability issues**: Identify and fix the causes of extreme RMSE values in certain configurations.

3. **Refine continuity constraints**: Reevaluate the continuity constraints to determine why they're not significantly improving reconstruction quality.

4. **Explore alternative reconstruction algorithms**: Investigate other approaches that might better capture the structure in the triad series.