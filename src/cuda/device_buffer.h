#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace sep {
namespace cuda {

/**
 * Forward declarations for device buffer utilities
 */

// Basic device buffer management functions
template <typename T>
void* allocateDeviceBuffer(size_t count);

template <typename T>
void freeDeviceBuffer(void* ptr);

template <typename T>
void copyToDevice(T* device_ptr, const T* host_ptr, size_t count);

template <typename T>
void copyFromDevice(T* host_ptr, const T* device_ptr, size_t count);

} // namespace cuda
} // namespace sep