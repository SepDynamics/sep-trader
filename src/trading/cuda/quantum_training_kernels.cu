// CRITICAL: For CUDA compilation, include ALL necessary headers early
#include <cstddef>
#include <array>
#include <functional>
#include <vector>

#ifdef SEP_USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

extern "C" {

// CUDA kernel for quantum pattern training
__global__ void quantum_pattern_training_kernel(
    const float* input_data,
    float* output_patterns,
    int data_size,
    int pattern_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < data_size) {
        // Quantum training computation
        output_patterns[idx] = input_data[idx] * 0.5f + 0.5f;
    }
}

// Host function to launch quantum training
void launch_quantum_training(
    const float* input_data,
    float* output_patterns,
    int data_size,
    int pattern_count
) {
    dim3 blockSize(256);
    dim3 gridSize((data_size + blockSize.x - 1) / blockSize.x);
    
    quantum_pattern_training_kernel<<<gridSize, blockSize>>>(
        input_data, output_patterns, data_size, pattern_count
    );
    
    cudaDeviceSynchronize();
}

} // extern "C"

#endif // SEP_USE_CUDA
