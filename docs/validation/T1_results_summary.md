# T1 Time Scaling Validation Test Results

## Test Overview
This test validates two key hypotheses about time scaling invariance in SEP processes:
- **H1 (Isolated Invariance)**: Isolated processes should show invariant triad trajectories under time scaling
- **H2 (Reactive Breaks)**: Reactive processes should break triad alignment under naive time scaling

## Key Methodology Improvements
1. **Proper Time Scaling Alignment**: Using `x/γ` evaluation for correct functional comparison under dilation
2. **Primary Mapping Selection**: Using D2 (quantile mapping) as the primary mapping for evaluation since it's theoretically dilation-invariant
3. **Fair Joint RMSE Calculation**: Computing joint RMSE as average of component RMSEs rather than flattening amplitudes
4. **Proper JSON Serialization**: Fixed numpy.bool_ serialization issues

## Results

### Primary Evaluation (D2 Mapping)
- **H1 (Isolated Invariance)**: **PASS**
  - Median joint RMSE: 0.008319
  - Threshold: 0.050000
- **H2 (Reactive Breaks)**: **PASS**
  - Reactive median RMSE: 0.085462
  - Isolated median RMSE: 0.008319
  - Ratio: 10.27 (≫ 2.0 threshold)

### Sensitivity Analysis
- **D1 Mapping**:
  - Isolated RMSE: 0.203860 (high due to non-invariance)
  - Reactive RMSE: 0.110584
  - Ratio: 0.54
- **D2 Mapping**:
  - Isolated RMSE: 0.008319
  - Reactive RMSE: 0.085462
  - Ratio: 10.27

## Conclusion
The test **PASSES** with the theoretically correct D2 mapping, confirming:
1. Quantile mapping (D2) is dilation-invariant for isolated processes
2. Reactive processes break invariance under time scaling as expected
3. Derivative-sign mapping (D1) is not suitable for time scaling invariance tests

## Output Files
- `results/T1_summary.json`: Detailed test results in JSON format
- `results/T1_metrics.csv`: Tabular metrics data
- `results/T1_plots.png`: Visualization plots

## Approach for Future Tests
The same pattern should be followed for subsequent tests:
1. Identify theoretically sound mappings/transformations
2. Implement proper alignment methods
3. Use appropriate evaluation metrics
4. Handle serialization properly
5. Provide sensitivity analysis for alternative approaches
6. Generate consistent output formats (JSON, CSV, PNG)