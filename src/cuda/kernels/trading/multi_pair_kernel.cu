#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common/cuda_common.h"
#include "multi_pair_kernel.cuh"

namespace sep::cuda::trading {

namespace {

/**
 * @brief CUDA kernel for multi-pair processing
 * 
 * Processes data from multiple currency pairs in parallel
 * 
 * @param pair_data Input data for multiple currency pairs
 * @param processed_signals Output array for processed signals
 * @param pair_count Number of currency pairs
 * @param data_per_pair Amount of data points per pair
 */
__global__ void multiPairProcessingKernel(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair
) {
    int pair_idx = blockIdx.x;
    int data_idx = threadIdx.x;
    
    if (pair_idx < pair_count && data_idx < data_per_pair) {
        int global_idx = pair_idx * data_per_pair + data_idx;
        processed_signals[global_idx] = pair_data[global_idx] * 0.9f + 0.1f;
    }
}

} // anonymous namespace

cudaError_t launchMultiPairProcessingKernel(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair
) {
    // Validate input parameters
    if (!pair_data || !processed_signals || pair_count <= 0 || data_per_pair <= 0) {
        return cudaErrorInvalidValue;
    }

    // Configure kernel launch parameters
    dim3 blockSize(data_per_pair);
    dim3 gridSize(pair_count);
    
    // Launch the kernel
    multiPairProcessingKernel<<<gridSize, blockSize>>>(
        pair_data, processed_signals, pair_count, data_per_pair
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