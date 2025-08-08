#pragma once
// CRITICAL: For CUDA compilation, apply comprehensive std::array protection
#include <array>

// CUDA-specific fix for std::array compatibility
// This header must be included before any standard library headers in CUDA files

// Force inclusion of array before anything else can corrupt it
#include <array>

// Immediately undefine any conflicting array macro
#ifdef array
#undef array
#endif

#ifdef __CUDACC__
#include <cstddef>
#include <type_traits>
#include <functional>

// Final cleanup for CUDA compilation
#ifdef array
#undef array
#endif
#endif
