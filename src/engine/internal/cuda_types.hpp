#ifndef SEP_ENGINE_CUDA_TYPES_HPP
#define SEP_ENGINE_CUDA_TYPES_HPP

#include <cstddef>

#ifdef SEP_USE_CUDA
// CUDA is available - use real CUDA types 
#include <cuda_runtime.h>
#else
// CUDA not available - define dummy types only when we really need them
// Check if CUDA types are already defined to avoid conflicts
#if !defined(CUDA_VERSION) && !defined(__CUDA_RUNTIME_H__)
typedef void* cudaStream_t;
typedef int cudaError_t;
typedef void* cudaEvent_t;
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int major, minor;
};

// Define CUDA constants in a way that won't conflict
#ifndef cudaSuccess
const int cudaSuccess = 0;
#endif
#ifndef cudaErrorInvalidValue  
const int cudaErrorInvalidValue = 1;
#endif
#ifndef cudaStreamDefault
const cudaStream_t cudaStreamDefault = nullptr;
#endif

// Dummy CUDA functions
inline cudaError_t cudaSetDevice(int device) { (void)device; return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* stream) { *stream = nullptr; return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t error) { return error == cudaSuccess ? "success" : "error"; }
#endif // !defined(CUDA_VERSION) && !defined(__CUDA_RUNTIME_H__)
#endif // SEP_USE_CUDA
#include <string>
#include <vector>
#include <memory>
#include "engine/internal/error_handler.h"

namespace sep::cuda {

struct CudaMetrics {
    size_t total_memory = 0;
    size_t used_memory = 0;
    float memory_utilization = 0.0f;
    float gpu_utilization = 0.0f;
};

class CudaCore {
public:
    struct Impl;
    CudaCore();
    ~CudaCore();
    static CudaCore& instance();

    Error initialize(int device_id);
    bool is_initialized() const;
    Error setDevice(int device);
    int getDeviceCount() const;
    Error getDeviceProperties(cudaDeviceProp& props, int device) const;
    Error getMemoryInfo(size_t& free, size_t& total) const;
    Error getLastError() const;
    std::string getErrorString(cudaError_t error) const;
    CudaMetrics getMetrics() const;
    Error updateMetrics();

private:
    std::unique_ptr<Impl> impl_;
};

}

#endif // SEP_ENGINE_CUDA_TYPES_HPP
