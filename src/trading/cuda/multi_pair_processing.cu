// CRITICAL: For CUDA compilation, include ALL necessary headers early
#include <cstddef>
#include <array>
#include <functional>
#include <vector>

#ifdef SEP_USE_CUDA
#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

__global__ void multi_pair_processing_kernel(
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

void launch_multi_pair_processing(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair
) {
    dim3 blockSize(data_per_pair);
    dim3 gridSize(pair_count);
    
    multi_pair_processing_kernel<<<gridSize, blockSize>>>(
        pair_data, processed_signals, pair_count, data_per_pair
    );
    
    cudaDeviceSynchronize();
}

} // extern "C"

#endif // SEP_USE_CUDA
