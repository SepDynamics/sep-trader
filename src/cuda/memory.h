#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace sep {
namespace cuda {

/**
 * RAII wrapper for device memory allocation
 */
template <typename T>
class DeviceMemory {
public:
    explicit DeviceMemory(size_t size);
    ~DeviceMemory();
    
    // Move constructor and assignment
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;
    
    // No copy operations
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Get the device pointer
    T* get() const;
    
    // Get the size in elements
    size_t size() const;
    
private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

// Memory management utilities
void getDeviceMemoryInfo(size_t& free, size_t& total);
void printDeviceMemoryStats();

// Pinned memory allocation
template <typename T>
T* allocatePinnedMemory(size_t count);

template <typename T>
void freePinnedMemory(T* ptr);

// Managed memory allocation
template <typename T>
T* allocateManagedMemory(size_t count);

template <typename T>
void freeManagedMemory(T* ptr);

// Memory prefetching
template <typename T>
void prefetchToDevice(T* ptr, size_t count, int device_id = 0);

template <typename T>
void prefetchToHost(T* ptr, size_t count);

// Memory utilities
template <typename T>
void zeroDeviceMemory(T* device_ptr, size_t count);

} // namespace cuda
} // namespace sep