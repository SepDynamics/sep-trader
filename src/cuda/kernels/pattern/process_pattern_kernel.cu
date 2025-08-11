#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "process_pattern_kernel.cuh"
#include "pattern_types.cuh"
#include "../../../cuda/common/error/cuda_error.h"
#include "../../../cuda/common/device_buffer.h"
#include "../../../cuda/common/stream.h"

namespace sep {
namespace cuda {
namespace pattern {

namespace {
__device__ unsigned int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}
} // anonymous namespace

// Helper device functions
__device__ float calculateCoherence(const PatternData& pattern) {
    // Ensure we have at least 4 attributes
    if (pattern.get_size() < 4) return 0.5f;
    
    float real_part = pattern[0];
    float imag_part = pattern[1];
    float amplitude = pattern[2];
    float phase = pattern[3];

    float coherence = amplitude * cosf(phase) * real_part + amplitude * sinf(phase) * imag_part;
    return fmaxf(0.1f, fminf(1.0f, coherence));
}

__device__ float calculateStability(const PatternData& pattern) {
    // Use the 5th attribute as stability, or default to 0.5
    return pattern.get_size() > 4 ? pattern[4] : 0.5f;
}

__device__ void evolvePattern(PatternData& pattern, float evolutionRate, float timeDelta) {
    // Ensure we have at least 4 attributes
    if (pattern.get_size() < 4) return;
    
    float mutation = evolutionRate * timeDelta * 1e-5f;
    float2 mutationVector = make_float2(cosf(mutation), sinf(mutation));

    float2 newState = make_float2(pattern[0] * mutationVector.x - pattern[1] * mutationVector.y,
                                  pattern[0] * mutationVector.y + pattern[1] * mutationVector.x);

    pattern[0] = newState.x;
    pattern[1] = newState.y;

    float magnitude = sqrtf((newState.x * newState.x) + (newState.y * newState.y));
    if (magnitude > 1e-5f) {
        pattern[0] /= magnitude;
        pattern[1] /= magnitude;
        pattern[2] = magnitude;
    }

    pattern[3] = atan2f(newState.y, newState.x);
}

__global__ void processPatternKernel(PatternData* patterns, PatternData* results,
                                    size_t patternCount, const PatternData* previousPatterns) {
    const unsigned int idx = get_global_idx();
    if (idx >= patternCount)
        return;

    PatternData pattern = patterns[idx];

    // PatternData doesn't have a coherence member, store in attributes if needed
    float coherence = calculateCoherence(pattern);

    float timeDelta = 0.016f;
    float evolutionRate = 0.05f;

    evolvePattern(pattern, evolutionRate, timeDelta);

    if (previousPatterns != nullptr && idx > 0 && idx < patternCount - 1) {
        PatternData leftPattern = patterns[idx - 1];
        PatternData rightPattern = patterns[idx + 1];

        float interactionStrength = 0.05f;
        if (pattern.get_size() >= 2 && leftPattern.get_size() >= 2 && rightPattern.get_size() >= 2) {
            pattern[0] += interactionStrength * (leftPattern[0] + rightPattern[0]) * 0.5f;
            pattern[1] += interactionStrength * (leftPattern[1] + rightPattern[1]) * 0.5f;
        }
    }

    results[idx] = pattern;
}

// Host-side launcher function
cudaError_t launchProcessPatternKernel(
    PatternData* h_patterns, 
    PatternData* h_results,
    size_t patternCount, 
    const PatternData* h_previousPatterns,
    cudaStream_t stream) 
{
    // For PatternData, we need to manually handle memory allocation
    // since it contains a pointer to float data
    PatternData* d_patterns = nullptr;
    PatternData* d_results = nullptr;
    PatternData* d_previousPatterns = nullptr;

    // Allocate device memory for pattern arrays
    CUDA_CHECK(cudaMalloc(&d_patterns, patternCount * sizeof(PatternData)));
    CUDA_CHECK(cudaMalloc(&d_results, patternCount * sizeof(PatternData)));
    
    if (h_previousPatterns) {
        CUDA_CHECK(cudaMalloc(&d_previousPatterns, patternCount * sizeof(PatternData)));
    }

    // For each pattern, allocate device memory for its data
    for (size_t i = 0; i < patternCount; i++) {
        // Allocate and copy pattern data
        float* d_data = nullptr;
        size_t data_size = h_patterns[i].size * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_data, data_size));
        CUDA_CHECK(cudaMemcpy(d_data, h_patterns[i].data, data_size, cudaMemcpyHostToDevice));
        
        // Create a temporary host PatternData with device pointers
        PatternData temp_pattern;
        temp_pattern.data = d_data;
        temp_pattern.size = h_patterns[i].size;
        
        // Copy the temporary pattern to device
        CUDA_CHECK(cudaMemcpy(&d_patterns[i], &temp_pattern, sizeof(PatternData), cudaMemcpyHostToDevice));
    }

    // Copy previous patterns if provided
    if (h_previousPatterns) {
        for (size_t i = 0; i < patternCount; i++) {
            float* d_data = nullptr;
            size_t data_size = h_previousPatterns[i].size * sizeof(float);
            
            CUDA_CHECK(cudaMalloc(&d_data, data_size));
            CUDA_CHECK(cudaMemcpy(d_data, h_previousPatterns[i].data, data_size, cudaMemcpyHostToDevice));
            
            PatternData temp_pattern;
            temp_pattern.data = d_data;
            temp_pattern.size = h_previousPatterns[i].size;
            
            CUDA_CHECK(cudaMemcpy(&d_previousPatterns[i], &temp_pattern, sizeof(PatternData), cudaMemcpyHostToDevice));
        }
    }
    
    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((patternCount + blockSize.x - 1) / blockSize.x);
    
    processPatternKernel<<<gridSize, blockSize, 0, stream>>>(
        d_patterns,
        d_results,
        patternCount,
        h_previousPatterns ? d_previousPatterns : nullptr
    );
    
    // Check for kernel launch errors
    CUDA_CHECK_LAST();
    
    // Copy results back to host
    for (size_t i = 0; i < patternCount; i++) {
        // Get the device pattern data pointer
        PatternData device_pattern;
        CUDA_CHECK(cudaMemcpy(&device_pattern, &d_results[i], sizeof(PatternData), cudaMemcpyDeviceToHost));
        
        // Copy the pattern data back to host
        CUDA_CHECK(cudaMemcpy(h_results[i].data, device_pattern.data, h_results[i].size * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // Free device memory for pattern data
    for (size_t i = 0; i < patternCount; i++) {
        PatternData device_pattern;
        CUDA_CHECK(cudaMemcpy(&device_pattern, &d_patterns[i], sizeof(PatternData), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(device_pattern.data));
        
        CUDA_CHECK(cudaMemcpy(&device_pattern, &d_results[i], sizeof(PatternData), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(device_pattern.data));
    }
    
    // Free previous patterns memory if allocated
    if (h_previousPatterns) {
        for (size_t i = 0; i < patternCount; i++) {
            PatternData device_pattern;
            CUDA_CHECK(cudaMemcpy(&device_pattern, &d_previousPatterns[i], sizeof(PatternData), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(device_pattern.data));
        }
        CUDA_CHECK(cudaFree(d_previousPatterns));
    }
    
    // Free device memory for pattern arrays
    CUDA_CHECK(cudaFree(d_patterns));
    CUDA_CHECK(cudaFree(d_results));
    
    // Synchronize stream
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    return cudaSuccess;
}

} // namespace pattern
} // namespace cuda
} // namespace sep