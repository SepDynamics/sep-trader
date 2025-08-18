# CUDA Development Guide

This document provides a comprehensive overview of the CUDA architecture, API design, and development practices for the SEP Engine.

## 1. Overview & Build Status

The CUDA components of the SEP Engine have been consolidated into a unified library located at `src/cuda/`. This library provides a consistent, type-safe, and high-performance interface for all GPU-accelerated operations.

**Build Status:** The `sep_cuda` library compiles successfully with the main project build system. All kernels are correctly found and compiled, and the common infrastructure for error handling and memory management is in place.

## 2. CUDA API Design Principles

The CUDA API is designed with the following principles in mind:

- **Consistency:** Uniform patterns for naming, parameters, and error handling.
- **Type Safety:** Use of templates and strong typing to prevent errors at compile-time.
- **RAII (Resource Acquisition Is Initialization):** Automatic resource management for all CUDA objects (memory, streams, etc.) to prevent leaks.
- **Performance:** Optimized for high-throughput, low-overhead operations.
- **Testability:** APIs are designed to be mockable and testable in isolation.

### 2.1. Error Handling

A centralized error-handling mechanism is used across the library. The `CUDA_CHECK` macro provides a convenient way to wrap all CUDA runtime calls and throw a detailed exception upon failure.

```cpp
// Location: src/cuda/common/error.h
#define CUDA_CHECK(expr) \
  do { \
    cudaError_t status = (expr); \
    if (status != cudaSuccess) { \
      throw CudaException(CheckCudaStatus(status), \
                          "CUDA error: " #expr); \
    } \
  } while(0)
```

### 2.2. Memory Management

RAII wrappers are provided for all CUDA memory operations to ensure safety and prevent memory leaks.

- **`DeviceMemory<T>`:** Manages device-side memory (`cudaMalloc`/`cudaFree`).
- **`HostMemory<T>`:** Manages host-side page-locked (pinned) memory for high-speed asynchronous transfers.

```cpp
// Location: src/cuda/common/memory.h

// Example Usage
sep::cuda::DeviceMemory<float> d_data(1024);
std::vector<float> h_data(1024);
d_data.copyToDevice(h_data.data(), 1024);
```

### 2.3. Stream Management

The `sep::cuda::Stream` class is an RAII wrapper for CUDA streams, enabling asynchronous kernel execution and memory transfers.

## 3. Kernel Inventory

All CUDA kernels are organized by domain within the `src/cuda/kernels/` directory.

### 3.1. Quantum Processing Kernels
*Location: `src/cuda/kernels/quantum/`*

- **QBSA (Quantum Binary State Analysis):** `qbsa_kernel`
- **QSH (Quantum State Hierarchy):** `qsh_kernel`
- **QFH (Quantum Field Harmonics):** `quantum_fourier_hierarchy_kernel`
- **And others:** for similarity, state evolution, coherence, etc.

### 3.2. Pattern Processing Kernels
*Location: `src/cuda/kernels/pattern/`*

- **Bit Pattern Operations:** Kernels for compression, expansion, comparison, and transformation of bit patterns.
- **Pattern Analysis:** Kernels for evolution, coherence, stability, and matching.

### 3.3. Trading Computation Kernels
*Location: `src/cuda/kernels/trading/`*

- **Multi-Pair Processing:** Kernels for analyzing correlations and patterns across multiple currency pairs.
- **Pattern Analysis:** Kernels for identifying candlestick patterns, trends, and volatility.
- **Model Training:** Kernels for training quantum models, optimizing parameters, and backtesting.

## 4. Development and Verification

### 4.1. Adding New Kernels

1.  **Place the Kernel:** Add your new `.cu` and `.cuh` files to the appropriate subdirectory within `src/cuda/kernels/`.
2.  **Update CMake:** Add your new source file to the `sep_cuda` library target in `src/cuda/CMakeLists.txt`.
3.  **Follow API Design:** Use the established RAII wrappers and error-handling macros.

### 4.2. Verification

A standalone `cuda_verify_test` was historically used to confirm the CUDA build and runtime configuration. While removed to clean up the codebase, it serves as a good template for creating new verification tests.

**To re-create a verification test:**
1.  Create a new test `.cpp` file in a relevant `tests/` directory.
2.  In the test, initialize the CUDA device, prepare host data, and call the kernel you wish to test.
3.  Verify that the results returned to the host are correct.
4.  Add the new test executable to the corresponding `CMakeLists.txt`.

```cpp

```