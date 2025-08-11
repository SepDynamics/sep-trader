#include "pattern_analysis_kernel.cuh"
#include "../../common/cuda_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace sep::cuda::trading {

namespace {

/**
 * @brief CUDA kernel for pattern analysis
 * 
 * Analyzes market data patterns
 * 
 * @param market_data Input market data for analysis
 * @param analysis_results Output array for analysis results
 * @param data_points Number of data points to analyze
 */
__global__ void patternAnalysisKernel(
    const float* market_data,
    float* analysis_results,
    int data_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < data_points) {
        analysis_results[idx] = market_data[idx] * 0.8f;
    }
}

} // anonymous namespace

cudaError_t launchPatternAnalysisKernel(
    const float* market_data,
    float* analysis_results,
    int data_points
) {
    // Validate input parameters
    if (!market_data || !analysis_results || data_points <= 0) {
        return cudaErrorInvalidValue;
    }

    // Configure kernel launch parameters
    // Use 256 threads per block for optimal occupancy
    dim3 blockSize(256);
    dim3 gridSize((data_points + blockSize.x - 1) / blockSize.x);
    
    // Launch the kernel
    patternAnalysisKernel<<<gridSize, blockSize>>>(
        market_data, analysis_results, data_points
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