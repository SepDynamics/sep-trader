// CUDA error handling implementation
#include "cuda_error.h"
#include <iostream>
#include <cstdio>

namespace sep {
namespace cuda {

// Implementation of additional error handling utilities that aren't inline in the header

void setupCudaErrorHandling() {
    // Initialize CUDA runtime with error checking
    cudaError_t error = cudaFree(0);
    if (error != cudaSuccess) {
        throw CudaException("Failed to initialize CUDA runtime", error);
    }
    
    // Additional setup can be performed here
    std::cout << "CUDA error handling initialized" << std::endl;
}

} // namespace cuda
} // namespace sep