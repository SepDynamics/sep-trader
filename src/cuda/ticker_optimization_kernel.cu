#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_common.h"
#include "cuda/ticker_optimization_kernel.cuh"
#include "trading_kernels.cuh"

namespace sep::cuda::trading {

cudaError_t launchTickerOptimizationKernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count) {
    return sep::quantum::launchTickerOptimizationKernel(
        ticker_data, optimized_parameters, param_count);
}

} // namespace sep::cuda::trading
