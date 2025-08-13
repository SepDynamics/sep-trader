// Merged from: src/core/internal/cuda/error/cuda_error.cu
#include "cuda_error.cuh"

namespace sep::cuda::error {

namespace {
thread_local char error_buffer[1024];
}

CudaError::CudaError(cudaError_t error, const char* file, int line)
    : std::runtime_error(createErrorMessage(error, file, line)) {}

const char* CudaError::createErrorMessage(cudaError_t error, const char* file, int line) {
    const char* errorStr = cudaGetErrorString(error);
    snprintf(error_buffer, sizeof(error_buffer), 
             "CUDA error at %s:%d: %s (code %d)", 
             file, line, errorStr, error);
    return error_buffer;
}

void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw CudaError(err, file, line);
    }
}

} // namespace sep::cuda::error

// Merged from: src/cuda/error/cuda_error.cu
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

// Note: The original file was mostly empty, so I've added a placeholder
// implementation to demonstrate how it could be expanded.
