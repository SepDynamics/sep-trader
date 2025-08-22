#pragma once

#include <cuda_runtime.h>
#include "trading_types.cuh"
#include "multi_pair_kernel.cuh"
#include "trading_kernels.cuh"
#include "cuda/ticker_optimization_kernel.cuh"

/**
 * @file trading_kernels.h
 * @brief Facade for all trading-related CUDA kernels
 * 
 * This header provides a unified interface to all trading-related CUDA kernels.
 * It abstracts away the implementation details and provides a clean, simple
 * interface for the rest of the codebase to use.
 */

namespace sep::cuda::trading {

/**
 * @brief Process multiple currency pairs in parallel
 * 
 * @param pair_data Input data for multiple currency pairs
 * @param processed_signals Output array for processed signals
 * @param pair_count Number of currency pairs
 * @param data_per_pair Amount of data points per pair
 * @return cudaError_t Error code
 */
inline cudaError_t processMultiPairs(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair
) {
    return launchMultiPairProcessingKernel(
        pair_data,
        processed_signals,
        pair_count,
        data_per_pair
    );
}

/**
 * @brief Analyze market data patterns
 * 
 * @param market_data Input market data for analysis
 * @param analysis_results Output array for analysis results
 * @param data_points Number of data points to analyze
 * @return cudaError_t Error code
 */
inline cudaError_t analyzePatterns(
    const float* market_data,
    float* analysis_results,
    int data_points
) {
    return sep::quantum::launchPatternAnalysisKernel(
        market_data,
        analysis_results,
        data_points
    );
}

/**
 * @brief Train quantum patterns for trading
 * 
 * @param input_data Input data for quantum pattern training
 * @param output_patterns Output array for the trained patterns
 * @param data_size Size of the input data
 * @param pattern_count Number of patterns to generate
 * @return cudaError_t Error code
 */
inline cudaError_t trainQuantumPatterns(
    const float* input_data,
    float* output_patterns,
    int data_size,
    int pattern_count
) {
    return sep::quantum::launchQuantumTrainingKernel(
        input_data,
        output_patterns,
        data_size,
        pattern_count
    );
}

/**
 * @brief Optimize ticker parameters for trading
 * 
 * @param ticker_data Input ticker data for optimization
 * @param optimized_parameters Output array for optimized parameters
 * @param param_count Number of parameters to optimize
 * @return cudaError_t Error code
 */
inline cudaError_t optimizeTicker(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count
) {
    return launchTickerOptimizationKernel(
        ticker_data,
        optimized_parameters,
        param_count
    );
}

} // namespace sep::cuda::trading