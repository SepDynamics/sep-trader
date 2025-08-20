#ifndef SEP_CUDA_KERNELS_H
#define SEP_CUDA_KERNELS_H

#include "cuda_runtime.h"
#include <stdexcept>
#include <string>

namespace sep {
namespace cuda {

template <typename Kernel, typename... Args>
void launch_kernel(Kernel kernel, const float* input_data, float* output_patterns, size_t data_size, int num_patterns) {
    float* d_input_data;
    float* d_output_patterns;

    cudaError_t err = cudaMalloc(&d_input_data, data_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for input data: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_output_patterns, num_patterns * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_input_data);
        throw std::runtime_error("Failed to allocate device memory for output patterns: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(d_input_data, input_data, data_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input_data);
        cudaFree(d_output_patterns);
        throw std::runtime_error("Failed to copy input data from host to device: " + std::string(cudaGetErrorString(err)));
    }

    int block_size = 256;
    int grid_size = (num_patterns + block_size - 1) / block_size;
    kernel<<<grid_size, block_size>>>(d_input_data, d_output_patterns, data_size, num_patterns);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input_data);
        cudaFree(d_output_patterns);
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(output_patterns, d_output_patterns, num_patterns * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input_data);
        cudaFree(d_output_patterns);
        throw std::runtime_error("Failed to copy output patterns from device to host: " + std::string(cudaGetErrorString(err)));
    }

    cudaFree(d_input_data);
    cudaFree(d_output_patterns);
}

} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_KERNELS_H