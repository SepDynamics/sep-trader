#ifndef SEP_CUDA_COMMON_H
#define SEP_CUDA_COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace sep {
namespace cuda {

// Version information
constexpr int CUDA_API_VERSION_MAJOR = 1;
constexpr int CUDA_API_VERSION_MINOR = 0;
constexpr int CUDA_API_VERSION_PATCH = 0;

// Common CUDA constants
constexpr int DEFAULT_BLOCK_SIZE_1D = 256;
constexpr int DEFAULT_BLOCK_SIZE_2D = 16;  // 16x16 = 256 threads
constexpr size_t DEFAULT_SHARED_MEMORY_SIZE = 0;

// Helper function declarations moved to kernel_launch.h

} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_COMMON_H