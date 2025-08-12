#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common/cuda_common.h"
#include "ticker_optimization_kernel.cuh"

namespace sep::cuda::trading {

namespace {

/**
 * @brief CUDA kernel for ticker optimization
 * 
 * Optimizes ticker parameters for trading
 * 
 * @param ticker_data Input ticker data for optimization
 * @param optimized_parameters Output array for optimized parameters
 * @param param_count Number of parameters to optimize
 */
__global__ void tickerOptimizationKernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < param_count) {
        optimized_parameters[idx] = ticker_data[idx] * 1.2f;
    }
}

} // anonymous namespace

cudaError_t launchTickerOptimizationKernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count
) {
    // Validate input parameters
    if (!ticker_data || !optimized_parameters || param_count <= 0) {
        return cudaErrorInvalidValue;
    }

    // Configure kernel launch parameters
    // Use 256 threads per block for optimal occupancy
    dim3 blockSize(256);
    dim3 gridSize((param_count + blockSize.x - 1) / blockSize.x);
    
    // Launch the kernel
    tickerOptimizationKernel<<<gridSize, blockSize>>>(
        ticker_data, optimized_parameters, param_count
    );
    
    // Check for asynchronous errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    // Synchronize and check for errors
    return cudaDeviceSynchronize();
}

} // namespace sep::cuda::trading