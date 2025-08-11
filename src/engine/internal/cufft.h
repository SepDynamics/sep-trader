#ifndef SEP_COMPAT_CUFFT_H
#define SEP_COMPAT_CUFFT_H

#include <cuda_runtime.h>


#if SEP_ENGINE_HAS_CUDA
#  if defined(__has_include) && __has_include(<cufft.h>)
#    include <cufft.h>
#    define SEP_HAS_CUFFT 1
#  else
#    define SEP_HAS_CUFFT 0
#  endif
#else
#  define SEP_HAS_CUFFT 0
#endif

#if !SEP_HAS_CUFFT
#  ifdef SEP_BACKTESTING
// Define stub types and functions for backtesting builds when cuFFT is absent
namespace sep {
namespace cuda {

// CUFFT Types
typedef int cufftResult;
typedef int cufftHandle;
typedef int cufftType;

} // namespace cuda
} // namespace sep
#  else
#    error "cuFFT required for production builds. Enable SEP_BACKTESTING for stubs."
#  endif
#endif

#endif // SEP_COMPAT_CUFFT_H