#include <cuda_runtime.h>
#include <cstdint>

#include <iostream>
#include <stdexcept>
#include <cstdint>

#include "core/cuda_error.cuh"
#include "device_buffer.h"
#include "memory.h"

namespace sep {
namespace cuda {

// RAII wrapper implementation
template <typename T>
DeviceMemory<T>::DeviceMemory(size_t size) : size_(size) {
    if (size > 0) {
        cudaError_t err = cudaMalloc(&ptr_, size_ * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "Failed to allocate device memory: " +
                std::string(cudaGetErrorString(err)));
        }
    }
}

template <typename T>
DeviceMemory<T>::~DeviceMemory() {
    if (ptr_) {
        cudaFree(ptr_);
    }
}

template <typename T>
DeviceMemory<T>::DeviceMemory(DeviceMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

template <typename T>
DeviceMemory<T>& DeviceMemory<T>::operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_) {
            cudaFree(ptr_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

template <typename T>
T* DeviceMemory<T>::get() const {
    return ptr_;
}

template <typename T>
size_t DeviceMemory<T>::size() const {
    return size_;
}

// Device memory utilities
void getDeviceMemoryInfo(size_t& free, size_t& total) {
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
}

void printDeviceMemoryStats() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    std::cout << "CUDA Memory Usage:" << std::endl;
    std::cout << "  Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Free memory:  " << (free_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Used memory:  " << ((total_mem - free_mem) / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Usage:        " << ((total_mem - free_mem) * 100.0 / total_mem) << "%" << std::endl;
}

// Pinned memory
template <typename T>
T* allocatePinnedMemory(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
void freePinnedMemory(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

// Managed memory
template <typename T>
T* allocateManagedMemory(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
void freeManagedMemory(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

template <typename T>
void prefetchToDevice(T* ptr, size_t count, int device_id) {
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, count * sizeof(T), device_id));
}

template <typename T>
void prefetchToHost(T* ptr, size_t count) {
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, count * sizeof(T), cudaCpuDeviceId));
}

template <typename T>
void zeroDeviceMemory(T* device_ptr, size_t count) {
    CUDA_CHECK(cudaMemset(device_ptr, 0, count * sizeof(T)));
}

// Explicit template instantiations
template class DeviceMemory<float>;
template class DeviceMemory<double>;
template class DeviceMemory<int>;
template class DeviceMemory<char>;

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

