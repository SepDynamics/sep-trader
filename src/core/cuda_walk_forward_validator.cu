// Minimal CUDA includes to avoid conflicts
#define _DISABLE_FPCLASSIFY_FUNCTIONS 1
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS 1
#include <cuda_runtime.h>

// Include our header after CUDA headers
#include "core/cuda_walk_forward_validator.hpp"

// Forward declarations of used functions
extern "C" {
    int printf(const char*, ...);
}

namespace sep::validation::cuda
{
    __global__ void walk_forward_kernel(
        const float* input_data, float* output_data,
        size_t data_size, size_t window_size, size_t num_windows)
    {
        extern __shared__ float shared_mem[];

        const int tid = threadIdx.x;
        const int gid = blockIdx.x * blockDim.x + tid;

        if (gid >= num_windows) {
            return;
        }

        const int start_index = gid;
        const int end_index = start_index + window_size;

        if (end_index > data_size) {
            return;
        }

        float sum = 0.0f;
        for (int i = start_index; i < end_index; ++i) {
            sum += input_data[i];
        }

        shared_mem[tid] = sum;
        __syncthreads();

        // Reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_mem[tid] += shared_mem[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output_data[blockIdx.x] = shared_mem[0] / window_size;
        }
    }

    __global__ void volumetric_sampling_kernel(
        const float* market_data, float* volume_output,
        size_t data_points, size_t depth_levels, size_t time_buckets)
    {
        // Kernel implementation will be added here
    }

    __global__ void coherence_calculation_kernel(
        const float* data1, const float* data2, float* coherence_output,
        size_t data_size)
    {
        // Kernel implementation will be added here
    }

    extern "C" {
        void cuda_walk_forward_kernel(
            const float* input_data, float* output_data,
            size_t data_size, size_t window_size, size_t num_windows,
            size_t threads_per_block, size_t num_blocks)
        {
            const dim3 blockSize(threads_per_block);
            const dim3 gridSize(num_blocks);
            walk_forward_kernel<<<gridSize, blockSize>>>(
                input_data, output_data, data_size, window_size, num_windows);
        }

        void cuda_volumetric_sampling_kernel(
            const float* market_data, float* volume_output,
            size_t data_points, size_t depth_levels, size_t time_buckets,
            size_t threads_per_block, size_t num_blocks)
        {
            const dim3 blockSize(threads_per_block);
            const dim3 gridSize(num_blocks);
            volumetric_sampling_kernel<<<gridSize, blockSize>>>(
                market_data, volume_output, data_points, depth_levels, time_buckets);
        }

        void cuda_coherence_calculation_kernel(
            const float* data1, const float* data2, float* coherence_output,
            size_t data_size, size_t threads_per_block, size_t num_blocks)
        {
            const dim3 blockSize(threads_per_block);
            const dim3 gridSize(num_blocks);
            coherence_calculation_kernel<<<gridSize, blockSize>>>(
                data1, data2, coherence_output, data_size);
        }
    }
}