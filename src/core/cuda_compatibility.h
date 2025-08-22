#pragma once

#include "cuda_prerequisites.h"
#include "cuda_type_system.h"

// Cross-compilation semantics and runtime dispatch
namespace cuda {
namespace compatibility {

// Host-device function attributes
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#define CUDA_DEVICE_ONLY __device__
#define CUDA_HOST_ONLY __host__
#else
#define CUDA_CALLABLE
#define CUDA_DEVICE_ONLY
#define CUDA_HOST_ONLY
#endif

// Runtime dispatch helpers
template<typename T>
struct RuntimeDispatch {
    template<typename Func>
    static auto dispatch(Func&& f) {
        #ifdef __CUDACC__
        // CUDA context - use device dispatch
        return f.template operator()<cuda::ts::vector>();
        #else
        // Host context - use standard dispatch
        return f.template operator()<std::vector>();
        #endif
    }
};

// Type trait helpers for CUDA compatibility
template<typename T>
struct is_cuda_compatible : std::false_type {};

// Vector specialization
template<typename T>
struct is_cuda_compatible<cuda::ts::vector<T>> : std::true_type {};

// String specialization
template<>
struct is_cuda_compatible<cuda::ts::string> : std::true_type {};

template<typename T>
inline constexpr bool is_cuda_compatible_v = is_cuda_compatible<T>::value;

// Namespace management
namespace detail {
    using namespace cuda::ts;
} // namespace detail

} // namespace compatibility

// Convenience namespace alias
namespace compat = compatibility;

} // namespace cuda