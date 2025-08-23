// Disable fpclassify functions that cause conflicts with CUDA internal headers
#define _DISABLE_FPCLASSIFY_FUNCTIONS 1
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS 1

#include "cuda/trading_kernels.cuh"

namespace sep { namespace quantum {

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

cudaError_t launchPatternAnalysisKernel(const float* market_data, float* analysis_results, int data_points) {
    return launchScaleBias(market_data, analysis_results, data_points, 0.8f, 0.0f);
}

cudaError_t launchQuantumTrainingKernel(const float* input_data, float* output_patterns, int data_size, int /*pattern_count*/) {
    return launchScaleBias(input_data, output_patterns, data_size, 0.5f, 0.5f);
}

namespace {
__global__ void multiPairProcessingKernel(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair) {
    int pair_idx = blockIdx.x;
    int data_idx = threadIdx.x;
    if (pair_idx < pair_count && data_idx < data_per_pair) {
        int global_idx = pair_idx * data_per_pair + data_idx;
        processed_signals[global_idx] = pair_data[global_idx] * 0.9f + 0.1f;
    }
}

inline cudaError_t launchMultiPairProcessing(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair) {
    if (!pair_data || !processed_signals || pair_count <= 0 || data_per_pair <= 0) {
        return cudaErrorInvalidValue;
    }
    dim3 blockSize(data_per_pair);
    dim3 gridSize(pair_count);
    multiPairProcessingKernel<<<gridSize, blockSize>>>(
        pair_data, processed_signals, pair_count, data_per_pair);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaDeviceSynchronize();
}
} // anonymous namespace

cudaError_t launchMultiPairProcessingKernel(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair) {
    return launchMultiPairProcessing(pair_data, processed_signals, pair_count, data_per_pair);
}

cudaError_t launchTickerOptimizationKernel(
    const float* ticker_data,
    float* optimized_parameters,
    int param_count) {
    return launchScaleBias(ticker_data, optimized_parameters, param_count, 1.2f, 0.0f);
}

}} // namespace sep::quantum

