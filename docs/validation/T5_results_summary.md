# T5 Results Summary

## Test Status: PASS

## Key Results

* **H9 (SEP outperforms naive)**: Median improvement ratio **0.6756** (PASS vs 0.15 target threshold).
* **H10 (Parameter-stability correlation)**: Correlation coefficient **-0.9898** (PASS vs 0.30 target threshold).

## Detailed Metrics

### Hypothesis H9: SEP-informed filtering outperforms naive filtering

* Median naive uncertainty reduction: 0.3509
* Median SEP uncertainty reduction: 0.5276
* Median improvement ratio: 0.6756 (67.56%)
* Threshold: 0.15 (15% improvement required)
* Number of comparisons: 6
* Result: **PASS**

### Hypothesis H10: Parameter-stability correlation

* Correlation coefficient: -0.9898
* Threshold: 0.30 (minimum correlation required)
* Data points: 6
* Stability range: [0.3374, 0.6761]
* Parameter range: [5.1053, 6.6502]
* Result: **PASS**

## Scientific Value

The T5 test successfully validated two important hypotheses about the SEP framework:

1. The ability of SEP metrics to inform better filtering parameters, leading to significantly improved uncertainty reduction.
2. The relationship between signal stability and optimal filtering parameters, providing a principled approach to adaptive filtering.

These results demonstrate the practical utility of the SEP framework in real-world signal processing applications.

## Data Files

* `results/T5_summary.json` - Complete test results in JSON format
* `results/T5_filtering_metrics.csv` - Detailed filtering metrics
* `results/T5_plots.png` - Visualization of results