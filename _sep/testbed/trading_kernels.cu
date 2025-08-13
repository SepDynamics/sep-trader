#include <cuda_runtime.h>

namespace sep { namespace testbed {

namespace {
__global__ void scaleBiasKernel(const float* input, float* output, int n, float scale, float bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale + bias;
    }
}

inline cudaError_t launchScaleBias(const float* input, float* output, int n, float scale, float bias) {
    if (!input || !output || n <= 0) return cudaErrorInvalidValue;
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    scaleBiasKernel<<<gridSize, blockSize>>>(input, output, n, scale, bias);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}
} // anonymous namespace

cudaError_t analyzePatterns(const float* market_data, float* analysis_results, int data_points) {
    return launchScaleBias(market_data, analysis_results, data_points, 0.8f, 0.0f);
}

cudaError_t trainQuantumPatterns(const float* input_data, float* output_patterns, int data_size, int /*pattern_count*/) {
    return launchScaleBias(input_data, output_patterns, data_size, 0.5f, 0.5f);
}

}} // namespace sep::testbed

