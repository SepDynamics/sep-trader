#include <cuda_runtime.h>

#include <cstdint>

#include "common/error/cuda_error.h"
#include "qsh_kernel.cuh"

namespace sep {
namespace cuda {
namespace quantum {

namespace {

__device__ std::uint32_t derivativeCascade(std::uint64_t input, std::uint32_t cascade_depth) {
    std::uint32_t result = 0;
    for (std::uint32_t i = 0; i < cascade_depth; ++i) {
        std::uint32_t xor_result = static_cast<std::uint32_t>(input) ^ static_cast<std::uint32_t>(input >> 32);
        result ^= xor_result;
        input = static_cast<std::uint64_t>(xor_result) | (static_cast<std::uint64_t>(xor_result) << 32);
    }
    return result;
}

} // anonymous namespace

__global__ void qsh_kernel(
    const std::uint64_t* d_chunks,
    std::uint32_t num_chunks,
    std::uint32_t* d_collapse_indices,
    std::uint32_t* d_collapse_counts
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_chunks)
        return;

    const std::uint64_t chunk = d_chunks[tid];
    std::uint32_t collapse_count = 0;

    std::uint32_t cascade_result = derivativeCascade(chunk, 3);

    const std::uint64_t reversed = __brevll(chunk);
    const std::uint64_t diff = chunk ^ reversed;

    const std::uint32_t pairs = 32;
    const std::uint64_t pair_mask = (1ULL << pairs) - 1ULL;

    std::uint32_t mismatches = 0;
    std::uint32_t current_run = 0;
    std::uint32_t max_run = 0;
    for (std::uint32_t i = 0; i < pairs; ++i) {
        bool mis = (diff >> i) & 1ULL;
        if (mis) {
            mismatches++;
            current_run++;
            if (current_run > max_run)
                max_run = current_run;
        } else {
            current_run = 0;
        }
    }

    const float mismatch_ratio = static_cast<float>(mismatches) / static_cast<float>(pairs);

    float cascade_factor = static_cast<float>(__popc(cascade_result)) / 32.0f;
    float adjusted_threshold = 0.35f * (1.0f - 0.2f * cascade_factor);
    bool rupture = (mismatch_ratio > adjusted_threshold) && (max_run > 2);

    std::uint32_t match_mask = static_cast<std::uint32_t>(~diff & pair_mask);
    const std::uint32_t base_idx = tid * pairs;

    while (match_mask && collapse_count < pairs) {
        std::uint32_t i = __ffs(match_mask) - 1;
        d_collapse_indices[base_idx + collapse_count] = i;
        collapse_count++;
        match_mask &= match_mask - 1;
    }

    if (rupture && collapse_count < pairs) {
        const std::uint32_t base_idx = tid * pairs;
        d_collapse_indices[base_idx + collapse_count] = 0xFFFFFFFFU;
        collapse_count++;
    }

    d_collapse_counts[tid] = collapse_count;
}

cudaError_t launchQSHKernel(
    const DeviceBuffer<std::uint64_t>& chunks,
    DeviceBuffer<std::uint32_t>& collapse_indices,
    DeviceBuffer<std::uint32_t>& collapse_counts,
    const Stream& stream
) {
    // Validate input parameters
    const std::uint32_t num_chunks = chunks.size();
    const std::uint32_t pairs = 32;
    
    // Ensure collapse_indices has enough space (pairs per chunk)
    if (collapse_indices.size() < num_chunks * pairs) {
        return cudaErrorInvalidValue;
    }
    
    // Ensure collapse_counts has enough space (one per chunk)
    if (collapse_counts.size() < num_chunks) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const std::uint32_t block_size = 256;
    const std::uint32_t grid_size = (num_chunks + block_size - 1) / block_size;
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    qsh_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        chunks.data(),
        num_chunks,
        collapse_indices.data(),
        collapse_counts.data()
    );
    
    return cudaGetLastError();
}

}}} // namespace sep::cuda::quantum