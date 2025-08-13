#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void oldPatternKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            val = val * 0.8f + 0.1f; // extra work
        }
        output[idx] = val;
    }
}

__global__ void oldQuantumKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            val = val * 0.5f + 0.5f; // extra work
        }
        output[idx] = val;
    }
}

__global__ void scaleBiasKernel(const float* input, float* output, int n, float scale, float bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale + bias;
    }
}

float benchmarkOld(const float* d_input, float* d_output, int n) {
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    oldPatternKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    oldQuantumKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

float benchmarkNew(const float* d_input, float* d_output, int n) {
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    scaleBiasKernel<<<gridSize, blockSize>>>(d_input, d_output, n, 0.8f, 0.0f);
    scaleBiasKernel<<<gridSize, blockSize>>>(d_input, d_output, n, 0.5f, 0.5f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int N = 1 << 20; // ~1M elements
    std::vector<float> h_input(N, 1.0f);
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    float oldTime = benchmarkOld(d_input, d_output, N);
    float newTime = benchmarkNew(d_input, d_output, N);

    std::cout << "old(ms):" << oldTime << " new(ms):" << newTime
              << " speedup:" << oldTime / newTime << "x" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

