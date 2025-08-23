#include <cuda_runtime.h>
// Removed <cmath> include to fix fpclassify errors
#include <stdexcept>
#include <string>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ float calculate_pattern_coherence(const float* price_data, size_t data_size, int pattern_idx) {
    // Bit-transition harmonic analysis implementation
    float coherence = 0.0f;
    float phase_accumulator = 0.0f;
    
    // Apply quantum field harmonic transforms
    for (size_t i = 1; i < data_size; ++i) {
        float price_change = price_data[i] - price_data[i-1];
        float normalized_change = tanhf(price_change * 1000.0f); // Scale for forex precision
        
        // Harmonic phase calculation with bit-level analysis
        float bit_pattern = 0.0f;
        uint32_t price_bits = __float_as_uint(normalized_change);
        
        // Analyze bit transitions for harmonic patterns
        for (int bit = 0; bit < 32; ++bit) {
            if ((price_bits >> bit) & 1) {
                bit_pattern += sinf(2.0f * M_PI * bit / 32.0f * (pattern_idx + 1));
            }
        }
        
        phase_accumulator += bit_pattern * normalized_change;
        coherence += cosf(phase_accumulator) * expf(-0.1f * i / data_size); // Decay factor
    }
    
    return tanhf(coherence / sqrtf((float)data_size));
}

__global__ void quantum_training_kernel(const float* input_data, float* output_patterns, size_t data_size, int num_patterns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_patterns) {
        // Enhanced bit-transition harmonic analysis
        float pattern_strength = calculate_pattern_coherence(input_data, data_size, idx);
        
        // Apply quantum superposition analysis
        float superposition_factor = 1.0f;
        if (data_size > 1) {
            // Calculate price volatility for superposition weighting
            float volatility = 0.0f;
            for (size_t i = 1; i < data_size; ++i) {
                float change = input_data[i] - input_data[i-1];
                volatility += change * change;
            }
            volatility = sqrtf(volatility / (data_size - 1));
            superposition_factor = 1.0f / (1.0f + expf(-volatility * 10000.0f)); // Sigmoid activation
        }
        
        // Combine pattern coherence with superposition analysis
        float quantum_signal = pattern_strength * superposition_factor;
        
        // Apply stability threshold - only output high-confidence predictions
        float confidence_threshold = 0.15f; // Minimum confidence for trading signals
        if (fabsf(quantum_signal) > confidence_threshold) {
            output_patterns[idx] = quantum_signal;
        } else {
            output_patterns[idx] = 0.0f; // No signal if confidence too low
        }
    }
}

extern "C" void launch_quantum_training(const float* input_data, float* output_patterns, size_t data_size, int num_patterns) {
    // Direct CUDA kernel launch implementation
    float* d_input_data;
    float* d_output_patterns;

    cudaError_t err = cudaMalloc(&d_input_data, data_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for input data: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_output_patterns, num_patterns * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_input_data);
        throw std::runtime_error("Failed to allocate device memory for output patterns: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(d_input_data, input_data, data_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input_data);
        cudaFree(d_output_patterns);
        throw std::runtime_error("Failed to copy input data from host to device: " + std::string(cudaGetErrorString(err)));
    }

    int block_size = 256;
    int grid_size = (num_patterns + block_size - 1) / block_size;
    quantum_training_kernel<<<grid_size, block_size>>>(d_input_data, d_output_patterns, data_size, num_patterns);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input_data);
        cudaFree(d_output_patterns);
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(output_patterns, d_output_patterns, num_patterns * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input_data);
        cudaFree(d_output_patterns);
        throw std::runtime_error("Failed to copy output patterns from device to host: " + std::string(cudaGetErrorString(err)));
    }

    cudaFree(d_input_data);
    cudaFree(d_output_patterns);
}
