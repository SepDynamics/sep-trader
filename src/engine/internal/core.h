#pragma once

// This preprocessor directive __CUDACC__ is defined ONLY by the nvcc compiler.
// This ensures your mock types are only included for standard g++ compilation,
// and NEVER when the real CUDA headers are being used.
#ifndef __CUDACC__
    #include "cuda_types.hpp"
#endif

// Now, include the real CUDA headers. They will be used by nvcc and
// by g++ in files that need the official runtime API.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>

namespace sep::cuda {

}
