#include "pattern_kernels.cuh"

namespace sep::cuda::trading {
namespace {
__global__ void patternAnalysisKernelOptimized(
    const float* market_data,
    float* analysis_results,
    int data_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 4;
    if (base < data_points) {
        #pragma unroll
        for (int j = 0; j < 4 && base + j < data_points; ++j) {
            analysis_results[base + j] = market_data[base + j] * 0.8f;
        }
    }
}
} // anonymous namespace

cudaError_t launchPatternAnalysisKernel(
    const float* market_data,
    float* analysis_results,
    int data_points) {
    if (!market_data || !analysis_results || data_points <= 0) {
        return cudaErrorInvalidValue;
    }
    dim3 blockSize(256);
    dim3 gridSize((data_points + blockSize.x * 4 - 1) / (blockSize.x * 4));
    patternAnalysisKernelOptimized<<<gridSize, blockSize>>>(
        market_data, analysis_results, data_points);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    return cudaDeviceSynchronize();
}

} // namespace sep::cuda::trading
