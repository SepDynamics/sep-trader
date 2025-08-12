#ifndef SEP_CUDA_QSH_KERNEL_CUH
#define SEP_CUDA_QSH_KERNEL_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "common/kernel_launch.h"
#include "common/memory/device_buffer.h"
#include "common/stream/stream.h"

namespace sep {
namespace cuda {
namespace quantum {

// QSH (Quantum State Hierarchy) kernel function
// Performs pattern collapse analysis using derivative cascades
__global__ void qsh_kernel(
    const std::uint64_t* d_chunks,
    std::uint32_t num_chunks,
    std::uint32_t* d_collapse_indices,
    std::uint32_t* d_collapse_counts
);

// QSH kernel launch wrapper using Buffer abstractions
cudaError_t launchQSHKernel(
    const DeviceBuffer<std::uint64_t>& chunks,
    DeviceBuffer<std::uint32_t>& collapse_indices,
    DeviceBuffer<std::uint32_t>& collapse_counts,
    const Stream& stream = Stream()
);

}}} // namespace sep::cuda::quantum

#endif // SEP_CUDA_QSH_KERNEL_CUH