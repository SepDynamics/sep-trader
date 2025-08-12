#include "kernel_launch.h"
#include "error/cuda_error.h"

namespace sep {
namespace cuda {

// Implementation of the kernel launch template function
// Must be in a .cu file to support CUDA triple-angle bracket syntax
template <typename KernelFunc, typename... Args>
void launchKernel(const LaunchConfig& config, KernelFunc kernel, Args&&... args) {
    kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        std::forward<Args>(args)...
    );
    CUDA_CHECK_LAST();
}

// Explicit instantiations for common kernel function pointer types would go here
// This would need to be customized based on actual kernel signatures used in the project

} // namespace cuda
} // namespace sep