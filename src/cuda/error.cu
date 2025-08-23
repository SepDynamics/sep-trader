// Disable fpclassify functions that cause conflicts with CUDA internal headers
#define _DISABLE_FPCLASSIFY_FUNCTIONS 1
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS 1

#include "core/cuda_error.cuh"
#include <cstdio>

namespace sep {
namespace cuda {

// Implementation of additional error handling utilities that aren't inline in the header


} // namespace cuda
} // namespace sep