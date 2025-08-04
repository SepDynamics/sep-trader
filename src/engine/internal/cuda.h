#pragma once

#ifdef __CUDACC__
#ifdef SEP_USE_CUDA
#include <cuda_runtime.h>
#endif
#include "engine/internal/cuda_helpers.h"
#include "engine/internal/cuda_sep.h"
#endif
#include "engine/internal/core.h"

namespace sep::cuda {

// CUDA utility functions and definitions
class CudaCore;

} // namespace sep::cuda
