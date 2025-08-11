# CUDA API Design Specification

## Overview

This document outlines the API design for the consolidated CUDA library in the SEP Engine. The API is designed to provide a consistent, type-safe, and RAII-compliant interface for CUDA operations across all components of the system.

## Design Principles

1. **Consistency** - Establish uniform patterns for function naming, parameter ordering, and error handling
2. **Type Safety** - Utilize templates and strong typing to catch errors at compile time
3. **Resource Management** - Implement RAII patterns for all CUDA resources
4. **Performance** - Optimize for high-throughput CUDA operations with minimal CPU overhead
5. **Testability** - Design APIs to be mockable and testable in isolation

## Common Components

### 1. Error Handling

All CUDA functions should use a consistent error handling mechanism:

```cpp
// In cuda/common/error.h
namespace sep {
namespace cuda {

enum class Status {
  Success = 0,
  InvalidArgument,
  OutOfMemory,
  KernelLaunchFailure,
  DeviceNotFound,
  NotImplemented,
  // Additional error codes...
};

// Exception class for CUDA errors
class CudaException : public std::exception {
public:
  explicit CudaException(Status status, const char* message = nullptr);
  Status status() const;
  const char* what() const noexcept override;
  // ...
};

// Utility to check CUDA runtime API status
Status CheckCudaStatus(cudaError_t status);

// Macro for error checking with line/file information
#define CUDA_CHECK(expr) \
  do { \
    cudaError_t status = (expr); \
    if (status != cudaSuccess) { \
      throw CudaException(CheckCudaStatus(status), \
                          "CUDA error: " #expr); \
    } \
  } while(0)

}} // namespace sep::cuda
```

### 2. Memory Management

RAII wrappers for CUDA memory management:

```cpp
// In cuda/common/memory.h
namespace sep {
namespace cuda {

// Device memory RAII wrapper
template <typename T>
class DeviceMemory {
public:
  // Allocate device memory of size count
  explicit DeviceMemory(size_t count);
  
  // Free device memory on destruction
  ~DeviceMemory();
  
  // Move semantics
  DeviceMemory(DeviceMemory&& other) noexcept;
  DeviceMemory& operator=(DeviceMemory&& other) noexcept;
  
  // No copy semantics
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
  
  // Access methods
  T* get() const;
  size_t size() const;
  
  // Copy data to device
  void copyToDevice(const T* host_data, size_t count);
  
  // Copy data from device
  void copyToHost(T* host_data, size_t count) const;
  
private:
  T* data_;
  size_t count_;
};

// Host memory RAII wrapper with page-locked (pinned) memory support
template <typename T>
class HostMemory {
public:
  enum class AllocationType {
    Standard,
    Pinned,
    Mapped
  };
  
  // Allocate host memory of size count with specified allocation type
  explicit HostMemory(size_t count, AllocationType type = AllocationType::Standard);
  
  // Free host memory on destruction
  ~HostMemory();
  
  // Move semantics
  HostMemory(HostMemory&& other) noexcept;
  HostMemory& operator=(HostMemory&& other) noexcept;
  
  // No copy semantics
  HostMemory(const HostMemory&) = delete;
  HostMemory& operator=(const HostMemory&) = delete;
  
  // Access methods
  T* get() const;
  size_t size() const;
  
private:
  T* data_;
  size_t count_;
  AllocationType type_;
};

// Structure-of-Arrays (SoA) memory layout manager
template <typename... Types>
class SoAMemoryLayout {
  // Implementation details to follow
};

}} // namespace sep::cuda
```

### 3. Stream Management

RAII wrapper for CUDA streams:

```cpp
// In cuda/common/stream.h
namespace sep {
namespace cuda {

class Stream {
public:
  // Create a new CUDA stream
  Stream();
  
  // Destroy the CUDA stream
  ~Stream();
  
  // Move semantics
  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;
  
  // No copy semantics
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  
  // Get the underlying CUDA stream
  cudaStream_t get() const;
  
  // Synchronize the stream
  void synchronize();
  
private:
  cudaStream_t stream_;
};

}} // namespace sep::cuda
```

### 4. Kernel Launch Utilities

Type-safe kernel launch wrappers:

```cpp
// In cuda/common/kernel_launch.h
namespace sep {
namespace cuda {

// Type-safe kernel launch configuration
struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t shared_memory_bytes;
  Stream* stream;
  
  // Constructor with sensible defaults
  LaunchConfig(dim3 grid_dim, dim3 block_dim,
              size_t shared_mem = 0,
              Stream* stream_ptr = nullptr);
};

// Type-safe kernel launch function
template <typename KernelFunc, typename... Args>
void launchKernel(const LaunchConfig& config, KernelFunc kernel, Args&&... args);

}} // namespace sep::cuda
```

## Domain-Specific APIs

### 1. Core CUDA Functionality

```cpp
// In cuda/core/device.h
namespace sep {
namespace cuda {
namespace core {

// Device information and capabilities
struct DeviceProperties {
  int device_id;
  std::string name;
  size_t total_memory;
  int compute_capability_major;
  int compute_capability_minor;
  int multi_processor_count;
  // Additional properties...
};

// Get properties for the specified device
DeviceProperties getDeviceProperties(int device_id = 0);

// Select the best device based on criteria
int selectBestDevice();

// Set the current device
void setDevice(int device_id);

// Get the current device
int getCurrentDevice();

}}} // namespace sep::cuda::core
```

### 2. Quantum Processing APIs

```cpp
// In cuda/quantum/qbsa.h
namespace sep {
namespace cuda {
namespace quantum {

// QBSA (Quantum Binary State Analysis) API
class QBSAProcessor {
public:
  // Configuration for QBSA processing
  struct Config {
    float coherence_threshold;
    int max_iterations;
    // Additional parameters...
  };
  
  // Initialize the processor
  explicit QBSAProcessor(const Config& config);
  
  // Process quantum states
  Status processQuantumStates(const DeviceMemory<float>& input_states,
                             DeviceMemory<float>& output_states,
                             Stream* stream = nullptr);
  
  // Additional methods...
};

}}} // namespace sep::cuda::quantum
```

### 3. Pattern Processing APIs

```cpp
// In cuda/pattern/analyzer.h
namespace sep {
namespace cuda {
namespace pattern {

// Pattern analysis API
class PatternAnalyzer {
public:
  // Configuration for pattern analysis
  struct Config {
    int pattern_size;
    float stability_threshold;
    // Additional parameters...
  };
  
  // Initialize the analyzer
  explicit PatternAnalyzer(const Config& config);
  
  // Analyze patterns
  Status analyzePatterns(const DeviceMemory<float>& input_data,
                        DeviceMemory<float>& pattern_metrics,
                        Stream* stream = nullptr);
  
  // Additional methods...
};

}}} // namespace sep::cuda::pattern
```

### 4. Trading-Specific APIs

```cpp
// In cuda/trading/optimizer.h
namespace sep {
namespace cuda {
namespace trading {

// Trading optimization API
class TradingOptimizer {
public:
  // Configuration for trading optimization
  struct Config {
    int window_size;
    float risk_tolerance;
    // Additional parameters...
  };
  
  // Initialize the optimizer
  explicit TradingOptimizer(const Config& config);
  
  // Optimize trading parameters
  Status optimizeParameters(const DeviceMemory<float>& market_data,
                           DeviceMemory<float>& optimized_params,
                           Stream* stream = nullptr);
  
  // Additional methods...
};

}}} // namespace sep::cuda::trading
```

## CMake Integration

The consolidated CUDA library will be integrated into the build system with the following CMake structure:

```cmake
# In src/cuda/CMakeLists.txt
add_library(sep_cuda
  # Common
  common/error.cu
  common/memory.cu
  common/stream.cu
  common/kernel_launch.cu
  
  # Core
  core/device.cu
  
  # Quantum
  quantum/qbsa.cu
  
  # Pattern
  pattern/analyzer.cu
  
  # Trading
  trading/optimizer.cu
  
  # Additional files...
)

target_include_directories(sep_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sep_cuda PUBLIC ${CUDA_LIBRARIES})
```

## Next Steps

1. Implement the common components (error handling, memory management, stream management)
2. Create the core CUDA functionality implementation
3. Migrate existing quantum processing implementations to the new API
4. Migrate existing pattern processing implementations to the new API
5. Migrate existing trading-specific implementations to the new API
6. Develop comprehensive tests for each component
7. Update the rest of the codebase to use the new consolidated API
