#pragma once

// CUDA-safe includes that avoid std::array conflicts
// Always include array first before anything else that might need it
#include <array>

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

// Core C++ includes
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

// Ensure proper namespace resolution for CUDA context
namespace cuda {
namespace type_system {

// String type compatibility
using string = ::std::string;

// IO stream compatibility
namespace io {
    using ostream = ::std::ostream;
    using istream = ::std::istream;
    extern ostream& cout;
    extern ostream& cerr;
    extern ostream& (*endl)(ostream&);
} // namespace io

// Smart pointer compatibility
template<typename T, typename... Args>
using unique_ptr = ::std::unique_ptr<T, Args...>;



// Container compatibility
template<typename T>
using vector = ::std::vector<T>;

template<typename T>
using queue = ::std::queue<T>;

} // namespace type_system

// Convenience namespace alias
namespace ts = type_system;

} // namespace cuda

// Import CUDA-compatible types into global namespace
using cuda::ts::string;
namespace io = cuda::ts::io;
using io::cout;
using io::cerr;
using io::endl;

