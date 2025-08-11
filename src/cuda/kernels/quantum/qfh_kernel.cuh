#ifndef SEP_CUDA_QFH_KERNEL_CUH
#define SEP_CUDA_QFH_KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>

#include "../../common/kernel_launch.h"
#include "../../common/stream/stream.h"
#include "../../common/memory/device_buffer.h"

// Forward declaration
namespace sep {
namespace quantum {
namespace bitspace {
    struct ForwardWindowResult;
}}}

namespace sep {
namespace cuda {
namespace quantum {

// QFH (Quantum Fourier Hierarchy) kernel function
// Analyzes bit transitions to calculate damped coherence and stability
__global__ void qfh_kernel(
    const uint8_t* d_bit_packages, 
    int num_packages, 
    int package_size, 
    sep::quantum::bitspace::ForwardWindowResult* d_results
);

// QFH kernel launch wrapper using Buffer abstractions
cudaError_t launchQFHBitTransitionsKernel(
    const DeviceBuffer<uint8_t>& bit_packages,
    DeviceBuffer<sep::quantum::bitspace::ForwardWindowResult>& results,
    int package_size,
    const Stream& stream = Stream()
);

}}} // namespace sep::cuda::quantum

#endif // SEP_CUDA_QFH_KERNEL_CUH