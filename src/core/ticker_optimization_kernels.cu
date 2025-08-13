#include <array>

// CRITICAL: For CUDA compilation, include ALL necessary headers early
// #include <functional>  // Removed due to GCC 11 compatibility issues with CUDA
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cuda/ticker_optimization_kernel.cuh"

extern "C" {

void launch_ticker_optimization(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count
) {
    sep::cuda::trading::launchTickerOptimizationKernel(
        ticker_data, optimized_parameters, param_count
    );
}

} // extern "C"
