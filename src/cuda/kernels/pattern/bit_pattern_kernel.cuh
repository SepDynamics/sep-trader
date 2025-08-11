#ifndef SEP_CUDA_KERNELS_PATTERN_BIT_PATTERN_KERNEL_CUH
#define SEP_CUDA_KERNELS_PATTERN_BIT_PATTERN_KERNEL_CUH

#include <cuda_runtime.h>
#include "pattern_types.cuh"

namespace sep {
namespace cuda {
namespace pattern {

/**
 * @brief Analyzes bit patterns in a window to extract pattern metrics
 * 
 * This kernel processes a window of binary data (0s and 1s) to compute various
 * pattern metrics including:
 * - Flip count: Number of bit transitions in the window
 * - Rupture count: Number of consecutive 1s
 * - Entropy: Shannon entropy of the bit distribution
 * - Coherence: Pattern coherence metric
 * - Stability: Pattern stability metric
 * - Confidence: Confidence in the pattern analysis
 * 
 * @param d_bits Device pointer to binary data
 * @param total_bits_size Total size of the binary data array
 * @param index_start Starting index in the binary data array
 * @param window_size Size of the window to analyze
 * @param d_results Device pointer to results storage
 */
__global__ void analyzeBitPatternsKernel(
    const uint8_t* d_bits,
    size_t total_bits_size,
    size_t index_start,
    size_t window_size,
    sep::cuda::pattern::ForwardWindowResult* d_results
);

/**
 * @brief Launch the bit pattern analysis kernel
 * 
 * @param h_bits Host pointer to binary data
 * @param total_bits_size Total size of the binary data array
 * @param index_start Starting index in the binary data array
 * @param window_size Size of the window to analyze
 * @param h_results Host pointer to results storage
 * @param stream CUDA stream to use for asynchronous execution
 * @return cudaError_t Error code indicating success or failure
 */
cudaError_t launchAnalyzeBitPatternsKernel(
    const uint8_t* h_bits,
    size_t total_bits_size,
    size_t index_start,
    size_t window_size,
    sep::cuda::pattern::ForwardWindowResult* h_results,
    cudaStream_t stream
);

} // namespace pattern
} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_KERNELS_PATTERN_BIT_PATTERN_KERNEL_CUH