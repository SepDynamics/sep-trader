#pragma once

#include <cuda_runtime.h>
#include "trading_types.cuh"

namespace sep::quantum {

/**
 * @brief Launch the multi-pair processing kernel
 * 
 * @param pair_data Input data for multiple currency pairs
 * @param processed_signals Output processed signals
 * @param pair_count Number of currency pairs
 * @param data_per_pair Amount of data per pair
 * @return cudaError_t CUDA error code
 */
cudaError_t launchMultiPairProcessingKernel(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair);

/**
 * @brief Launch the ticker optimization kernel
 * 
 * @param ticker_data Input ticker data
 * @param optimized_parameters Output optimized parameters
 * @param param_count Number of parameters
 * @return cudaError_t CUDA error code
 */
cudaError_t launchTickerOptimizationKernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count);

} // namespace sep::quantum