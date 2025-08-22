#pragma once
#include <cuda_runtime.h>

namespace sep::quantum {

cudaError_t launchPatternAnalysisKernel(const float* market_data, float* analysis_results, int data_points);

cudaError_t launchQuantumTrainingKernel(const float* input_data, float* output_patterns, int data_size, int pattern_count);

cudaError_t launchMultiPairProcessingKernel(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair);

cudaError_t launchTickerOptimizationKernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count);

} // namespace sep::quantum

