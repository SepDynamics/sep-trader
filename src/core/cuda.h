#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

#include "cuda_helpers.h"
#include "cuda_sep.h"
#endif
#include "core.h"

namespace sep::cuda {

// CUDA utility functions and definitions
class CudaCore;

} // namespace sep::cuda
