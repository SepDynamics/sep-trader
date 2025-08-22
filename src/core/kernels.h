#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Forward declarations for CUDA kernel functions - Global scope for C linkage
extern "C" {
    // QBSA kernel launch function
    cudaError_t launchQBSAKernel(
        const std::uint32_t* d_probe_indices,
        const std::uint32_t* d_expectations,
        std::uint32_t num_probes,
        std::uint32_t* d_bitfield,
        std::uint32_t* d_corrections,
        std::uint32_t* d_correction_count,
        cudaStream_t stream = nullptr
    );

    // QSH kernel launch function
    cudaError_t launchQSHKernel(
        const std::uint64_t* d_chunks,
        std::uint32_t num_chunks,
        std::uint32_t* d_collapse_indices,
        std::uint32_t* d_collapse_counts,
        cudaStream_t stream = nullptr
    );
}

namespace sep {

namespace quantum {
namespace bitspace {
    // Forward declaration - actual definition is in qfh.h
    struct ForwardWindowResult;
}
}

} // namespace sep