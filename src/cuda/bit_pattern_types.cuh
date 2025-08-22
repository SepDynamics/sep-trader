#ifndef SEP_CUDA_PATTERN_BIT_PATTERN_TYPES_CUH
#define SEP_CUDA_PATTERN_BIT_PATTERN_TYPES_CUH

#include <cstdint>

namespace sep::apps::cuda {

// Device-side equivalent of ForwardWindowResult
struct ForwardWindowResultDevice {
    int flip_count;
    int rupture_count;
    float entropy;
    float coherence;
    float stability;
    float confidence;
};

} // namespace sep::apps::cuda

#endif // SEP_CUDA_PATTERN_BIT_PATTERN_TYPES_CUH
