#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "../../quantum/pattern_kernels.cuh"

extern "C" {

void launch_pattern_analysis(
    const float* market_data,
    float* analysis_results,
    int data_points) {
    sep::cuda::trading::launchPatternAnalysisKernel(
        market_data, analysis_results, data_points);
}

} // extern "C"
