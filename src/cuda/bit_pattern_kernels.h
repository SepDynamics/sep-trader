#ifndef SEP_CUDA_PATTERN_BIT_PATTERN_KERNELS_H
#define SEP_CUDA_PATTERN_BIT_PATTERN_KERNELS_H

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h> // For cudaStream_t

#include "result.h" // For sep::core::Result and SEPResult

#include "bit_pattern_types.cuh" // Include the new device types

namespace sep::apps::cuda {

using sep::SEPResult;

// Host-side launcher function for the bit pattern analysis kernel
extern "C" sep::core::Result launchAnalyzeBitPatternsKernel(const uint8_t* h_bits,
                                                      size_t total_bits_size,
                                                      size_t index_start,
                                                      size_t window_size,
                                                      ForwardWindowResultDevice* h_results,
                                                      cudaStream_t stream);

} // namespace sep::apps::cuda

#endif // SEP_CUDA_PATTERN_BIT_PATTERN_KERNELS_H
