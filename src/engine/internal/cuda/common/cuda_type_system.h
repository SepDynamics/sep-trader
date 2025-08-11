#pragma once

#include "cuda_prerequisites.h"

// CUDA-compatible type system bridge
namespace cuda {
namespace type_system {

// String type compatibility
#ifdef __CUDACC__
using string = ::std::string;
#else
using string = std::string;
#endif

// IO stream compatibility
#ifdef __CUDACC__
namespace io {
    using ostream = ::std::ostream;
    using istream = ::std::istream;
    using cout = ::std::cout;
    using cerr = ::std::cerr;
    using endl = ::std::endl;
} // namespace io
#else
namespace io {
    using std::ostream;
    using std::istream;
    using std::cout;
    using std::cerr;
    using std::endl;
} // namespace io
#endif

// Smart pointer compatibility
#ifdef __CUDACC__
template<typename T, typename... Args>
using unique_ptr = ::std::unique_ptr<T, Args...>;

template<typename T, typename... Args>
inline auto make_unique(Args&&... args) {
    return ::std::make_unique<T>(::std::forward<Args>(args)...);
}
#else
template<typename T, typename... Args>
using unique_ptr = std::unique_ptr<T, Args...>;

template<typename T, typename... Args>
inline auto make_unique(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}
#endif

// Container compatibility
#ifdef __CUDACC__
template<typename T>
using vector = ::std::vector<T>;

template<typename T>
using queue = ::std::queue<T>;
#else
template<typename T>
using vector = std::vector<T>;

template<typename T>
using queue = std::queue<T>;
#endif

} // namespace type_system

// Convenience namespace alias
namespace ts = type_system;

} // namespace cuda