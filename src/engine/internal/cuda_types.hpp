#ifndef SEP_ENGINE_CUDA_TYPES_HPP
#define SEP_ENGINE_CUDA_TYPES_HPP

#if defined(__CUDACC__) || defined(CUDA_VERSION)
// CUDA is available - use real CUDA types
#include <cuda_runtime.h>
#else
// Define CUDA types as dummies when CUDA is disabled
typedef void* cudaStream_t;
typedef int cudaError_t;
typedef void* cudaEvent_t;
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int major, minor;
};
#define cudaSuccess 0
#define cudaStreamDefault 0

// Dummy CUDA functions
inline cudaError_t cudaSetDevice(int device) { (void)device; return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* stream) { *stream = nullptr; return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t error) { return error == cudaSuccess ? "success" : "error"; }
#endif
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
