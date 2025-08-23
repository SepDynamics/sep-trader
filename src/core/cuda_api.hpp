#pragma once

#ifdef SEP_USE_CUDA
// Only include the real CUDA headers when CUDA support is enabled
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#else
#include <cstddef>
#include <cstdlib>

// Minimal stubs so CPU-only builds succeed without CUDA
using cudaStream_t = void*;
using cudaError_t = int;
using cudaMemcpyKind = int;
constexpr cudaError_t cudaSuccess = 0;

inline const char* cudaGetErrorString(cudaError_t) { return "CUDA disabled"; }
inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = std::malloc(size);
    return *ptr ? cudaSuccess : 1;
}
inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) {
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t) { return cudaSuccess; }
#endif
