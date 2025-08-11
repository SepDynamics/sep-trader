#pragma once

#include <cuda_runtime.h>
#include "trading_types.cuh"
#include "../../common/cuda_common.h"

namespace sep::cuda::trading {

/**
 * @brief Launch function for pattern analysis kernel
 * 
 * @param market_data Input market data for analysis
 * @param analysis_results Output array for analysis results
 * @param data_points Number of data points to analyze
 * @return cudaError_t Error code
 */
cudaError_t launchPatternAnalysisKernel(
    const float* market_data,
    float* analysis_results,
    int data_points
);

} // namespace sep::cuda::trading