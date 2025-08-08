#include "../../array_protection.h"

#ifdef SEP_USE_CUDA
#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

__global__ void ticker_optimization_kernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < param_count) {
        optimized_parameters[idx] = ticker_data[idx] * 1.2f;
    }
}

void launch_ticker_optimization(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count
) {
    dim3 blockSize(256);
    dim3 gridSize((param_count + blockSize.x - 1) / blockSize.x);
    
    ticker_optimization_kernel<<<gridSize, blockSize>>>(
        ticker_data, optimized_parameters, param_count
    );
    
    cudaDeviceSynchronize();
}

} // extern "C"

#endif // SEP_USE_CUDA
