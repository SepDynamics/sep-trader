#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "../../common/kernel_launch.h"
#include "../../common/stream/stream.h"
#include "../../common/memory/device_buffer.h"
#include "../../common/error/cuda_error.h"

#include "quantum_types.cuh"

namespace sep {
namespace cuda {
namespace quantum {

// Kernel for calculating pattern coherence in quantum space
__global__ void pattern_coherence_kernel(
    const float* d_pattern_data1,
    const float* d_pattern_data2,
    float* d_coherence_matrix,
    uint32_t pattern_size,
    uint32_t num_patterns
) {
    const uint32_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tid_x >= num_patterns || tid_y >= num_patterns)
        return;
    
    // Only compute upper triangle of coherence matrix (symmetry)
    if (tid_y > tid_x)
        return;
    
    float sum_product = 0.0f;
    float sum_squares1 = 0.0f;
    float sum_squares2 = 0.0f;
    
    // Compute dot product and magnitudes
    for (uint32_t i = 0; i < pattern_size; ++i) {
        const float val1 = d_pattern_data1[tid_x * pattern_size + i];
        const float val2 = d_pattern_data2[tid_y * pattern_size + i];
        
        sum_product += val1 * val2;
        sum_squares1 += val1 * val1;
        sum_squares2 += val2 * val2;
    }
    
    // Compute coherence (normalized dot product)
    float coherence = 0.0f;
    const float magnitude_product = sqrtf(sum_squares1 * sum_squares2);
    
    if (magnitude_product > 1e-6f) {
        coherence = sum_product / magnitude_product;
    }
    
    // Store coherence in matrix (both positions for symmetry)
    d_coherence_matrix[tid_y * num_patterns + tid_x] = coherence;
    if (tid_x != tid_y) {
        d_coherence_matrix[tid_x * num_patterns + tid_y] = coherence;
    }
}

// Kernel for calculating stability metrics from coherence values
__global__ void coherence_stability_kernel(
    const float* d_coherence_matrix,
    float* d_stability_scores,
    uint32_t num_patterns
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_patterns)
        return;
    
    float stability_sum = 0.0f;
    uint32_t count = 0;
    
    // Compute average coherence with other patterns
    for (uint32_t i = 0; i < num_patterns; ++i) {
        if (i == tid)
            continue;
        
        const float coherence = d_coherence_matrix[tid * num_patterns + i];
        stability_sum += coherence;
        count++;
    }
    
    // Store average stability score
    if (count > 0) {
        d_stability_scores[tid] = stability_sum / static_cast<float>(count);
    } else {
        d_stability_scores[tid] = 0.0f;
    }
}

// Launch wrapper for pattern coherence kernel
cudaError_t launchPatternCoherenceKernel(
    const DeviceBuffer<float>& pattern_data1,
    const DeviceBuffer<float>& pattern_data2,
    DeviceBuffer<float>& coherence_matrix,
    uint32_t pattern_size,
    uint32_t num_patterns,
    const Stream& stream
) {
    // Validate input parameters
    if (pattern_data1.size() != pattern_size * num_patterns ||
        pattern_data2.size() != pattern_size * num_patterns ||
        coherence_matrix.size() != num_patterns * num_patterns) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (num_patterns + block_size.x - 1) / block_size.x,
        (num_patterns + block_size.y - 1) / block_size.y
    );
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    pattern_coherence_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        pattern_data1.data(),
        pattern_data2.data(),
        coherence_matrix.data(),
        pattern_size,
        num_patterns
    );
    
    return cudaGetLastError();
}

// Launch wrapper for coherence stability kernel
cudaError_t launchCoherenceStabilityKernel(
    const DeviceBuffer<float>& coherence_matrix,
    DeviceBuffer<float>& stability_scores,
    uint32_t num_patterns,
    const Stream& stream
) {
    // Validate input parameters
    if (coherence_matrix.size() != num_patterns * num_patterns ||
        stability_scores.size() != num_patterns) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_patterns + block_size - 1) / block_size;
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    coherence_stability_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        coherence_matrix.data(),
        stability_scores.data(),
        num_patterns
    );
    
    return cudaGetLastError();
}

} // namespace quantum
} // namespace cuda
} // namespace sep