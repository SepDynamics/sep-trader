#pragma once

#ifndef SEP_CUDA_INCLUDES_CUH
#define SEP_CUDA_INCLUDES_CUH

// Use CUDA runtime host API headers
#if defined(__CUDACC__)
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
#else
    #include <cuda_runtime_api.h>  // Host-only header
#endif

// Standard C++ headers
#include <cstddef>

namespace sep::cuda {

// Common type definitions and constants
using CudaStreamHandle = cudaStream_t;
using CudaEventHandle = cudaEvent_t;

// Common CUDA constants
constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int MAX_GRID_SIZE = 65535;

} // namespace sep::cuda

#endif // SEP_CUDA_INCLUDES_CUH