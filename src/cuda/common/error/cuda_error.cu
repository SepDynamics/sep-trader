#include "cuda_error.h"
#include <iostream>

namespace sep {
namespace cuda {

// This file contains any non-inline implementations for cuda_error.h
// Most of the error handling functionality is already defined as inline in the header

// Additional error handling utilities could be added here
// For example, custom error callback registration, error logging, etc.

void setupCudaErrorHandling() {
    // Example of setting up a custom CUDA error handler
    // This is a placeholder for future enhancements
    
    // Could register a global CUDA error callback if needed
    // cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    
    // Initialize CUDA runtime with error checking
    cudaError_t error = cudaFree(0);
    if (error != cudaSuccess) {
        throw CudaException("Failed to initialize CUDA runtime", error);
    }
}

} // namespace cuda
} // namespace sep