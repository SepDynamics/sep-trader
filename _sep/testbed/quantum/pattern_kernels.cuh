#pragma once

#include <cuda_runtime.h>

namespace sep::cuda::trading {

cudaError_t launchPatternAnalysisKernel(
    const float* market_data,
    float* analysis_results,
    int data_points);

} // namespace sep::cuda::trading
