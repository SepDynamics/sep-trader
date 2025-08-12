#include <cuda_runtime.h>

#include <cstdint>

#include "common/error/cuda_error.h"
#include "qbsa_kernel.cuh"

namespace sep {
namespace cuda {
namespace quantum {

__global__ void qbsa_kernel(
    const std::uint32_t* d_probe_indices, 
    const std::uint32_t* d_expectations, 
    std::uint32_t num_probes,
    std::uint32_t* d_bitfield, 
    std::uint32_t* d_corrections, 
    std::uint32_t* d_correction_count
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probes)
        return;

    const std::uint32_t bit_index = d_probe_indices[tid];
    const std::uint32_t expected = d_expectations[tid];

    const std::uint32_t word_idx = bit_index / 32;
    const std::uint32_t bit_pos = bit_index % 32;
    const std::uint32_t bit_mask = 1U << bit_pos;

    const std::uint32_t current = atomicOr(&d_bitfield[word_idx], 0);
    const std::uint32_t current_bit = (current & bit_mask) ? 1 : 0;

    if (current_bit != expected) {
        atomicXor(&d_bitfield[word_idx], bit_mask);
        const std::uint32_t correction_idx = atomicAdd(d_correction_count, 1);
        d_corrections[correction_idx] = bit_index;
    }
}

cudaError_t launchQBSAKernel(
    const DeviceBuffer<std::uint32_t>& probe_indices,
    const DeviceBuffer<std::uint32_t>& expectations,
    DeviceBuffer<std::uint32_t>& bitfield,
    DeviceBuffer<std::uint32_t>& corrections,
    DeviceBuffer<std::uint32_t>& correction_count,
    const Stream& stream
) {
    // Validate input parameters
    const std::uint32_t num_probes = probe_indices.size();
    
    if (expectations.size() != num_probes) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const std::uint32_t block_size = 256;
    const std::uint32_t grid_size = (num_probes + block_size - 1) / block_size;
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    qbsa_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        probe_indices.data(),
        expectations.data(),
        num_probes,
        bitfield.data(),
        corrections.data(),
        correction_count.data()
    );
    
    return cudaGetLastError();
}

}}} // namespace sep::cuda::quantum