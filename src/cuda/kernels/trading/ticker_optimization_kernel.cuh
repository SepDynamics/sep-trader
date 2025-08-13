#pragma once

#include <cuda_runtime.h>

#include "../../common/cuda_common.h"
#include "trading_types.cuh"

namespace sep::cuda::trading {

/**
 * @brief Launch function for ticker optimization kernel
 * 
 * @param ticker_data Input ticker data for optimization
 * @param optimized_parameters Output array for optimized parameters
 * @param param_count Number of parameters to optimize
 * @return cudaError_t Error code
 */
cudaError_t launchTickerOptimizationKernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count
);

} // namespace sep::cuda::trading