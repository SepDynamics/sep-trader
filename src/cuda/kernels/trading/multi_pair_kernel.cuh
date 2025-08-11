#pragma once

#include <cuda_runtime.h>
#include "trading_types.cuh"
#include "../../common/cuda_common.h"

namespace sep::cuda::trading {

/**
 * @brief Launch function for multi-pair processing kernel
 * 
 * @param pair_data Input data for multiple currency pairs
 * @param processed_signals Output array for processed signals
 * @param pair_count Number of currency pairs
 * @param data_per_pair Amount of data points per pair
 * @return cudaError_t Error code
 */
cudaError_t launchMultiPairProcessingKernel(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair
);

} // namespace sep::cuda::trading