#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

#include "core/cuda_helpers.h"
#include "core/cuda_sep.h"
#endif
#include "core/core.h"

namespace sep::cuda {

// CUDA utility functions and definitions
class CudaCore;

} // namespace sep::cuda
