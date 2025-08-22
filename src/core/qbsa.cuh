#pragma once

#ifndef SRC_CORE_QBSA_MERGED_H
#define SRC_CORE_QBSA_MERGED_H

#include <cuda_runtime.h>
#include <cstdint>

// Originally from src/core/internal/qbsa.cuh
namespace sep::quantum {

struct QBSAParams {
    const uint32_t* probe_indices;
    const uint32_t* expectations;
    uint32_t* corrections;
    uint32_t num_probes;
    uint32_t max_corrections;
    float* correction_ratio;
};

// CUDA kernel declarations
__global__ void qbsa_kernel(QBSAParams params);

} // namespace sep::quantum

// Originally from src/core/qbsa.cuh
namespace sep::quantum::bitspace {

struct QBSAParams {
    const uint32_t* probe_indices;
    const uint32_t* expectations;
    uint32_t* corrections;
    uint32_t num_probes;
};

bool launch_qbsa_kernel(const QBSAParams& params);

} // namespace sep::quantum::bitspace

#endif // SRC_CORE_QBSA_MERGED_H
