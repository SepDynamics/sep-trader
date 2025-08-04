#pragma once

#ifdef __CUDACC__
#ifdef SEP_USE_CUDA
#include <cuda_runtime.h>
#endif

// CUDA math compatibility helpers
namespace sep::cuda {
    // Add any CUDA-specific math functions here if needed
}
#endif
