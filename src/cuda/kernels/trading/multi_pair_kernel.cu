#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common/cuda_common.h"
#include "multi_pair_kernel.cuh"
#include "quantum/trading_kernels.cuh"

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
