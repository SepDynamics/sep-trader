#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_common.h"
#include "cuda/ticker_optimization_kernel.cuh"
#include "trading_kernels.cuh"

namespace sep::cuda::trading {

__global__ void optimization_kernel(float* parameters, float* gradients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Simple optimization step
        parameters[idx] -= 0.01f * gradients[idx];
    }
}

extern "C" {

void launch_optimization_kernel(float* parameters, float* gradients, const float* ticker_data,
                                float* optimized_parameters, int param_count, int size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    optimization_kernel<<<grid, block>>>(parameters, gradients, size);
    cudaDeviceSynchronize();
}
}

cudaError_t launchTickerOptimizationKernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count) {
    return sep::quantum::launchTickerOptimizationKernel(
        ticker_data, optimized_parameters, param_count);
}

} // namespace sep::cuda::trading
