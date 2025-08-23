// Disable fpclassify functions that cause conflicts with CUDA internal headers
#define _DISABLE_FPCLASSIFY_FUNCTIONS 1
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS 1

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/cuda_common.h"
#include "cuda/multi_pair_kernel.cuh"
#include "cuda/trading_kernels.cuh"

namespace sep::cuda::trading {

cudaError_t launchMultiPairProcessingKernel(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair) {
    return sep::quantum::launchMultiPairProcessingKernel(
        pair_data, processed_signals, pair_count, data_per_pair);
}

} // namespace sep::cuda::trading
