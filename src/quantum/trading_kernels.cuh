#pragma once
#include <cuda_runtime.h>

namespace sep::quantum {

cudaError_t launchPatternAnalysisKernel(const float* market_data, float* analysis_results, int data_points);

cudaError_t launchQuantumTrainingKernel(const float* input_data, float* output_patterns, int data_size, int pattern_count);

} // namespace sep::quantum

