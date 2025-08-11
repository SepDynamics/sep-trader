#ifndef SEP_CUDA_PROCESS_PATTERN_KERNEL_CUH
#define SEP_CUDA_PROCESS_PATTERN_KERNEL_CUH

/**
 * @file process_pattern_kernel.cuh
 * @brief Declaration of CUDA kernels for pattern processing
 * 
 * This file declares the CUDA kernel and host-side launcher function
 * for processing patterns, including evolution and interaction logic.
 */

#include <cuda_runtime.h>
#include "pattern_types.cuh"

namespace sep {
namespace cuda {
namespace pattern {

/**
 * @brief CUDA kernel to process patterns
 * 
 * This kernel handles pattern evolution and interaction between adjacent patterns.
 * It applies quantum-based transformation rules to evolve patterns over time
 * and models interactions between neighboring patterns.
 * 
 * @param patterns Input pattern data array
 * @param results Output pattern data array
 * @param patternCount Number of patterns to process
 * @param previousPatterns Optional array of patterns from previous state (can be nullptr)
 */
__global__ void processPatternKernel(
    PatternData* patterns, 
    PatternData* results,
    size_t patternCount, 
    const PatternData* previousPatterns
);

/**
 * @brief Host-side launcher for the pattern processing kernel
 * 
 * This function handles memory allocation, data transfer, kernel launch,
 * and result retrieval for pattern processing.
 * 
 * @param patterns Host-side input pattern data array
 * @param results Host-side output pattern data array
 * @param patternCount Number of patterns to process
 * @param previousPatterns Optional array of patterns from previous state (can be nullptr)
 * @param stream CUDA stream to use for asynchronous execution
 * 
 * @return cudaError_t CUDA status code
 */
cudaError_t launchProcessPatternKernel(
    PatternData* patterns, 
    PatternData* results,
    size_t patternCount, 
    const PatternData* previousPatterns,
    cudaStream_t stream
);

} // namespace pattern
} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_PROCESS_PATTERN_KERNEL_CUH