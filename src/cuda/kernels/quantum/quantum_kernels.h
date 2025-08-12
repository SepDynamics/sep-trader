#ifndef SEP_CUDA_QUANTUM_KERNELS_H
#define SEP_CUDA_QUANTUM_KERNELS_H

#include <cuda_runtime.h>

#include <cstdint>

#include "common/memory/device_buffer.h"
#include "common/stream/stream.h"
#include "embedding_kernels.cuh"
#include "qbsa_kernel.cuh"
#include "qfh_kernel.cuh"
#include "qsh_kernel.cuh"
#include "quantum_types.cuh"

namespace sep {
namespace cuda {
namespace quantum {

// This header provides a unified interface to all quantum-related CUDA kernels
// Each section includes functions for specific quantum algorithm implementations

// Additional utility functions for quantum operations can be added here

}}} // namespace sep::cuda::quantum

#endif // SEP_CUDA_QUANTUM_KERNELS_H