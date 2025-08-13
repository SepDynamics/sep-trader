#include <array>

// CRITICAL: For CUDA compilation, include ALL necessary headers early
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "trading_kernels.cuh"

extern "C" {

void launch_multi_pair_processing(
    const float* pair_data,
    float* processed_signals,
    int pair_count,
    int data_per_pair
) {
    sep::quantum::launchMultiPairProcessingKernel(
        pair_data, processed_signals, pair_count, data_per_pair
    );
}

} // extern "C"
