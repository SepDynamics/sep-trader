#pragma once

#include <cuda_runtime.h>
#include "../../common/cuda_common.h"

namespace sep::cuda::trading {

// Common types for trading operations

/**
 * @brief Structure to hold multi-pair processing results
 */
struct MultiPairResult {
    float* processed_signals;
    int pair_count;
    int data_per_pair;
};

/**
 * @brief Structure to hold pattern analysis results
 */
struct PatternAnalysisResult {
    float* analysis_results;
    int data_points;
};

/**
 * @brief Structure to hold quantum training results
 */
struct QuantumTrainingResult {
    float* output_patterns;
    int data_size;
    int pattern_count;
};

/**
 * @brief Structure to hold ticker optimization results
 */
struct TickerOptimizationResult {
    float* optimized_parameters;
    int param_count;
};

} // namespace sep::cuda::trading