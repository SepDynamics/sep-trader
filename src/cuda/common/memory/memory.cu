#include <cuda_runtime.h>

#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "device_buffer.h"
#include "error/cuda_error.h"

namespace sep {
namespace cuda {

// CUDA Memory Management Utilities

// Get total and free device memory
void getDeviceMemoryInfo(size_t& free, size_t& total) {
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
}

// Print device memory usage statistics
void printDeviceMemoryStats() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    std::cout << "CUDA Memory Usage:" << std::endl;
    std::cout << "  Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Free memory:  " << (free_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Used memory:  " << ((total_mem - free_mem) / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Usage:        " << ((total_mem - free_mem) * 100.0 / total_mem) << "%" << std::endl;
}

// Allocate pinned (page-locked) host memory
template <typename T>
T* allocatePinnedMemory(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
    return ptr;
}

// Free pinned memory
template <typename T>
void freePinnedMemory(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

// Allocate managed memory (unified memory)
template <typename T>
T* allocateManagedMemory(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, count * sizeof(T)));
    return ptr;
}

// Free managed memory
template <typename T>
void freeManagedMemory(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Prefetch managed memory to device
template <typename T>
void prefetchToDevice(T* ptr, size_t count, int device_id = 0) {
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, count * sizeof(T), device_id));
}

// Prefetch managed memory to host
template <typename T>
void prefetchToHost(T* ptr, size_t count) {
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, count * sizeof(T), cudaCpuDeviceId));
}

// Zero memory on device
template <typename T>
void zeroDeviceMemory(T* device_ptr, size_t count) {
    CUDA_CHECK(cudaMemset(device_ptr, 0, count * sizeof(T)));
}

// Explicit template instantiations for common types
template uint32_t* allocatePinnedMemory<uint32_t>(size_t);
template void freePinnedMemory<uint32_t>(uint32_t*);
template uint32_t* allocateManagedMemory<uint32_t>(size_t);
template void freeManagedMemory<uint32_t>(uint32_t*);
template void prefetchToDevice<uint32_t>(uint32_t*, size_t, int);
template void prefetchToHost<uint32_t>(uint32_t*, size_t);
template void zeroDeviceMemory<uint32_t>(uint32_t*, size_t);

template float* allocatePinnedMemory<float>(size_t);
template void freePinnedMemory<float>(float*);
template float* allocateManagedMemory<float>(size_t);
template void freeManagedMemory<float>(float*);
template void prefetchToDevice<float>(float*, size_t, int);
template void prefetchToHost<float>(float*, size_t);
template void zeroDeviceMemory<float>(float*, size_t);

template double* allocatePinnedMemory<double>(size_t);
template void freePinnedMemory<double>(double*);
template double* allocateManagedMemory<double>(size_t);
template void freeManagedMemory<double>(double*);
template void prefetchToDevice<double>(double*, size_t, int);
template void prefetchToHost<double>(double*, size_t);
template void zeroDeviceMemory<double>(double*, size_t);

} // namespace cuda
} // namespace sep