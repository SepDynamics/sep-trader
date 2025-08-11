#ifndef SEP_CUDA_PATTERN_KERNELS_H
#define SEP_CUDA_PATTERN_KERNELS_H

/**
 * @file pattern_kernels.h
 * @brief Unified facade for pattern processing CUDA kernels
 * 
 * This header serves as a single inclusion point for all pattern-related CUDA kernel
 * functionality. It exposes a clean, consistent API for pattern analysis operations 
 * while hiding the implementation details of individual kernels.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include "bit_pattern_kernel.cuh"
#include "process_pattern_kernel.cuh"
#include "pattern_types.cuh"

namespace sep {
namespace cuda {
namespace pattern {

/**
 * @brief Analyzes bit patterns within a sliding window
 * 
 * This function provides a high-level interface to the bit pattern analysis kernel.
 * It processes a window of binary data to compute pattern metrics such as entropy,
 * coherence, stability, and confidence.
 * 
 * @param bits Pointer to the input bit array
 * @param total_size Total size of the bit array
 * @param start_index Starting index for the analysis window
 * @param window_size Size of the sliding window to analyze
 * @param result Pointer to a ForwardWindowResult struct to store the analysis results
 * @param stream CUDA stream to use for asynchronous execution (optional)
 * 
 * @return cudaError_t CUDA status code
 */
inline cudaError_t analyzeBitPatterns(
    const uint8_t* bits,
    size_t total_size,
    size_t start_index,
    size_t window_size,
    ForwardWindowResult* result,
    cudaStream_t stream = nullptr)
{
    return launchAnalyzeBitPatternsKernel(
        bits,
        total_size,
        start_index,
        window_size,
        result,
        stream
    );
}

/**
 * @brief Process and evolve patterns based on quantum transformation rules
 * 
 * This function provides a high-level interface to the pattern processing kernel.
 * It applies quantum-based transformation rules to evolve patterns over time and
 * models interactions between neighboring patterns. This is essential for quantum 
 * state evolution in the SEP engine's pattern processing pipeline.
 * 
 * @param patterns Pointer to the input pattern data array
 * @param results Pointer to the output pattern data array
 * @param patternCount Number of patterns to process
 * @param previousPatterns Optional array of patterns from previous state (can be nullptr)
 * @param stream CUDA stream to use for asynchronous execution (optional)
 * 
 * @return cudaError_t CUDA status code
 */
inline cudaError_t processPatterns(
    PatternData* patterns,
    PatternData* results,
    size_t patternCount,
    const PatternData* previousPatterns = nullptr,
    cudaStream_t stream = nullptr)
{
    return launchProcessPatternKernel(
        patterns,
        results,
        patternCount,
        previousPatterns,
        stream
    );
}

/**
 * @brief Batch-processes multiple windows of bit patterns
 * 
 * This function provides a high-level interface for analyzing multiple windows of
 * bit patterns in a single operation. It's optimized for scenarios where multiple
 * sliding windows need to be processed in parallel.
 * 
 * @param bits Pointer to the input bit array
 * @param total_size Total size of the bit array
 * @param start_indices Array of starting indices for each analysis window
 * @param window_sizes Array of window sizes for each analysis
 * @param num_windows Number of windows to analyze
 * @param results Pointer to an array of ForwardWindowResult structs to store the analysis results
 * @param stream CUDA stream to use for asynchronous execution (optional)
 * 
 * @return cudaError_t CUDA status code
 * 
 * @note This is a placeholder for future implementation. Currently not implemented.
 */
inline cudaError_t analyzeBitPatternsBatch(
    const uint8_t* bits,
    size_t total_size,
    const size_t* start_indices,
    const size_t* window_sizes,
    size_t num_windows,
    ForwardWindowResult* results,
    cudaStream_t stream = nullptr)
{
    // Placeholder for future implementation
    // This will be implemented when batch processing is needed
    return cudaErrorNotSupported;
}

// Additional pattern kernel functions will be added here as they are implemented
// Examples include:
// - classifyPatterns: Categorize patterns into predefined classes
// - detectPatternAnomalies: Identify statistical anomalies in patterns
// - mergePatterns: Combine multiple patterns using defined operations

} // namespace pattern
} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_PATTERN_KERNELS_H