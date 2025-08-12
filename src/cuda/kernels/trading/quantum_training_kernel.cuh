#pragma once

#include <cuda_runtime.h>

#include "common/cuda_common.h"
#include "trading_types.cuh"

namespace sep::cuda::trading {

/**
 * @brief Launch function for quantum pattern training kernel
 * 
 * @param input_data Input data for quantum pattern training
 * @param output_patterns Output array for the trained patterns
 * @param data_size Size of the input data
 * @param pattern_count Number of patterns to generate
 * @return cudaError_t Error code
 */
cudaError_t launchQuantumTrainingKernel(
    const float* input_data,
    float* output_patterns,
    int data_size,
    int pattern_count
);

} // namespace sep::cuda::trading