#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "common/error/cuda_error.h"
#include "qfh_kernel.cuh"
#include "quantum_types.cuh"

namespace sep {
namespace cuda {
namespace quantum {

__global__ void qfh_kernel(
    const uint8_t* d_bit_packages, 
    int num_packages, 
    int package_size, 
    sep::quantum::bitspace::ForwardWindowResult* d_results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_packages) {
        return;
    }

    const uint8_t* current_package = &d_bit_packages[idx * package_size];
    double accumulated_value = 0.0;
    double lambda = 0.1; // Decay constant

    for (int i = 1; i < package_size; ++i) {
        double future_bit = static_cast<double>(current_package[i]);
        double current_bit = static_cast<double>(current_package[0]);
        accumulated_value += (future_bit - current_bit) * exp(-lambda * i);
    }

    d_results[idx].damped_coherence = 1.0 - accumulated_value;
    d_results[idx].damped_stability = accumulated_value;
}

cudaError_t launchQFHBitTransitionsKernel(
    const DeviceBuffer<uint8_t>& bit_packages,
    DeviceBuffer<sep::quantum::bitspace::ForwardWindowResult>& results,
    int package_size,
    const Stream& stream
) {
    // Validate input parameters
    const int num_packages = bit_packages.size() / package_size;
    
    if (bit_packages.size() % package_size != 0) {
        return cudaErrorInvalidValue;
    }
    
    if (results.size() < static_cast<size_t>(num_packages)) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const int block_size = 256;
    const int grid_size = (num_packages + block_size - 1) / block_size;
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    qfh_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        bit_packages.data(),
        num_packages,
        package_size,
        results.data()
    );
    
    return cudaGetLastError();
}

}}} // namespace sep::cuda::quantum