#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

namespace sep {
namespace cuda {
namespace quantum {

// Common quantum data types and constants
constexpr uint32_t DEFAULT_THREAD_BLOCK_SIZE = 256;
constexpr float EPSILON = 1e-6f;

// Quantum state enumeration
enum class QuantumState : uint8_t {
    UNDEFINED = 0,
    UNSTABLE = 1,
    STABILIZING = 2,
    STABLE = 3,
    DESTABILIZING = 4,
    COLLAPSED = 5
};

// Pattern coherence threshold constants
struct CoherenceThresholds {
    static constexpr float UNSTABLE_THRESHOLD = 0.3f;
    static constexpr float STABILIZING_THRESHOLD = 0.5f;
    static constexpr float STABLE_THRESHOLD = 0.7f;
    static constexpr float DESTABILIZING_THRESHOLD = 0.6f;
    static constexpr float COLLAPSED_THRESHOLD = 0.2f;
};

// Structure for quantum binary state bit representation
struct QBSABitfield {
    uint32_t active_mask;       // Bits that are actively participating
    uint32_t state_bits;        // Current state bits
    uint32_t transition_mask;   // Bits in transition
    float stability_score;      // Stability metric (0.0-1.0)
    uint32_t generation_count;  // Number of generations/iterations
};

// Quantum pattern representation
struct QuantumPattern {
    uint32_t id;                // Unique pattern identifier
    uint32_t size;              // Pattern size in elements
    QuantumState state;         // Current quantum state
    float coherence_score;      // Pattern coherence metric
    float stability_score;      // Pattern stability metric
    uint32_t residence_time;    // Time in current state
    uint32_t generation_count;  // Pattern generation count
};

// Coherence matrix element descriptor
struct CoherenceElement {
    uint32_t pattern_id_1;      // First pattern ID
    uint32_t pattern_id_2;      // Second pattern ID
    float coherence_value;      // Coherence between patterns
    float stability_impact;     // Impact on stability
};

// QFH (Quantum Fourier Hierarchy) parameters
struct QFHParameters {
    uint32_t fft_size;          // Size of FFT
    uint32_t sampling_rate;     // Sampling rate for frequency domain
    uint32_t window_size;       // Analysis window size
    float min_frequency;        // Minimum frequency of interest
    float max_frequency;        // Maximum frequency of interest
    bool apply_window;          // Whether to apply window function
};

// Device function to determine quantum state from coherence
__device__ __forceinline__ QuantumState determineStateFromCoherence(float coherence) {
    if (coherence < CoherenceThresholds::UNSTABLE_THRESHOLD) {
        return QuantumState::UNSTABLE;
    } else if (coherence < CoherenceThresholds::STABILIZING_THRESHOLD) {
        return QuantumState::STABILIZING;
    } else if (coherence >= CoherenceThresholds::STABLE_THRESHOLD) {
        return QuantumState::STABLE;
    } else if (coherence >= CoherenceThresholds::DESTABILIZING_THRESHOLD) {
        return QuantumState::DESTABILIZING;
    } else {
        return QuantumState::COLLAPSED;
    }
}

// Device function to calculate magnitude of complex number
__device__ __forceinline__ float complexMagnitude(float real, float imag) {
    return sqrtf(real * real + imag * imag);
}

// Device function to calculate phase of complex number
__device__ __forceinline__ float complexPhase(float real, float imag) {
    return atan2f(imag, real);
}

// Utility struct for packing pattern with its coherence score
struct PatternCoherencePair {
    uint32_t pattern_id;
    float coherence_score;
};

} // namespace quantum
} // namespace cuda
} // namespace sep