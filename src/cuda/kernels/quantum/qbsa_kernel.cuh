#ifndef SEP_CUDA_QBSA_KERNEL_CUH
#define SEP_CUDA_QBSA_KERNEL_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "common/kernel_launch.h"
#include "common/memory/device_buffer.h"
#include "common/stream/stream.h"

namespace sep {
namespace cuda {
namespace quantum {

// QBSA (Quantum Binary State Analysis) kernel function
// Processes bit indices and makes corrections to bitfields
__global__ void qbsa_kernel(
    const std::uint32_t* d_probe_indices, 
    const std::uint32_t* d_expectations, 
    std::uint32_t num_probes,
    std::uint32_t* d_bitfield, 
    std::uint32_t* d_corrections, 
    std::uint32_t* d_correction_count
);

// QBSA kernel launch wrapper using Buffer abstractions
cudaError_t launchQBSAKernel(
    const DeviceBuffer<std::uint32_t>& probe_indices,
    const DeviceBuffer<std::uint32_t>& expectations,
    DeviceBuffer<std::uint32_t>& bitfield,
    DeviceBuffer<std::uint32_t>& corrections,
    DeviceBuffer<std::uint32_t>& correction_count,
    const Stream& stream = Stream()
);

}}} // namespace sep::cuda::quantum

#endif // SEP_CUDA_QBSA_KERNEL_CUH