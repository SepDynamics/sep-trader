#ifndef SEP_CUDA_QUANTUM_TYPES_CUH
#define SEP_CUDA_QUANTUM_TYPES_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace sep {
namespace quantum {
namespace bitspace {

// Forward window result structure used in QFH calculations
struct ForwardWindowResult {
    double damped_coherence;  // Coherence value with dampening applied
    double damped_stability;  // Stability value with dampening applied
};

}}} // namespace sep::quantum::bitspace

#endif // SEP_CUDA_QUANTUM_TYPES_CUH