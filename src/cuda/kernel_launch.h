#ifndef SEP_CUDA_KERNEL_LAUNCH_H
#define SEP_CUDA_KERNEL_LAUNCH_H

#include "../util/stable_headers.h"
#include <cuda_runtime.h>
#include "cuda_error.h"
#include "stream.h"

namespace sep {
namespace cuda {

// Type-safe kernel launch configuration
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_memory_bytes;
    cudaStream_t stream;
    
    // Constructor with sensible defaults
    LaunchConfig(
        dim3 grid_dim, 
        dim3 block_dim,
        size_t shared_mem = 0,
        cudaStream_t stream_ptr = nullptr
    ) : grid(grid_dim), 
        block(block_dim), 
        shared_memory_bytes(shared_mem), 
        stream(stream_ptr) {}
    
    // Convenience constructor for 1D grid/block
    LaunchConfig(
        unsigned int grid_dim, 
        unsigned int block_dim,
        size_t shared_mem = 0,
        cudaStream_t stream_ptr = nullptr
    ) : grid(grid_dim, 1, 1), 
        block(block_dim, 1, 1), 
        shared_memory_bytes(shared_mem), 
        stream(stream_ptr) {}
};

// NOTE: Actual kernel launch must be implemented in .cu files using:
//
// template <typename KernelFunc, typename... Args>
// void launchKernel(const LaunchConfig& config, KernelFunc kernel, Args&&... args) {
//     kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
//         std::forward<Args>(args)...
//     );
//     CUDA_CHECK_LAST();
// }

// Helper to calculate grid dimensions based on work size and block size
inline dim3 calculateGrid(dim3 work_size, dim3 block_size) {
    return dim3(
        (work_size.x + block_size.x - 1) / block_size.x,
        (work_size.y + block_size.y - 1) / block_size.y,
        (work_size.z + block_size.z - 1) / block_size.z
    );
}

// Convenience function for 1D work
inline dim3 calculateGrid(unsigned int work_size, unsigned int block_size) {
    return dim3((work_size + block_size - 1) / block_size, 1, 1);
}

} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_KERNEL_LAUNCH_H