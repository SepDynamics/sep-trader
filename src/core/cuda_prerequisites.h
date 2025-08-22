#pragma once

// CUDA compilation context setup
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_HOST
#endif

// Core C++ includes - ordered to avoid conflicts
#include <array>  // Must be first
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <queue>
#include <functional>
#include <type_traits>
#include <cstddef>
#include <stdexcept>

// For CUDA compilation, ensure we have proper array support
#ifdef __CUDACC__
// Verify std::array is available
static_assert(std::is_same_v<std::array<int, 1>, std::array<int, 1>>, "std::array must be available for CUDA");
#endif