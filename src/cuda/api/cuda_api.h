#ifndef SEP_CUDA_API_H
#define SEP_CUDA_API_H

// Include all public CUDA interfaces

// Common utilities
#include "common/cuda_common.h"
#include "common/error/cuda_error.h"
#include "common/kernel_launch.h"

// Memory management
#include "common/memory/buffer.h"
#include "common/memory/device_buffer.h"

// Stream management
#include "common/stream/stream.h"

namespace sep {
namespace cuda {

// Initialize CUDA subsystem
bool initialize(int device_id = 0);

// Check if CUDA is available
bool isAvailable();

// Get number of available CUDA devices
int getDeviceCount();

// Get current device ID
int getCurrentDevice();

// Set current device
void setDevice(int device_id);

// Get device properties
cudaDeviceProp getDeviceProperties(int device_id = -1);

// Reset device
void resetDevice();

// Synchronize device
void synchronizeDevice();

} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_API_H