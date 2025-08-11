#ifndef SEP_CUDA_KERNELS_PATTERN_TYPES_CUH
#define SEP_CUDA_KERNELS_PATTERN_TYPES_CUH

#include <cstdint>

namespace sep {
namespace cuda {
namespace pattern {

// Forward window result structure for bit pattern analysis
struct ForwardWindowResult {
    int flip_count;      // Count of bit flips in the window
    int rupture_count;   // Count of ruptures in the window
    float entropy;       // Shannon entropy of the pattern
    float coherence;     // Pattern coherence metric
    float stability;     // Pattern stability metric
    float confidence;    // Confidence in the pattern analysis
};

// Pattern data structure
struct PatternData {
    float* data;         // Pointer to pattern attributes
    size_t size;         // Number of attributes

    __device__ __host__ size_t get_size() const { return size; }
    
    __device__ __host__ float& operator[](size_t index) {
        return data[index];
    }
    
    __device__ __host__ const float& operator[](size_t index) const {
        return data[index];
    }
};

} // namespace pattern
} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_KERNELS_PATTERN_TYPES_CUH