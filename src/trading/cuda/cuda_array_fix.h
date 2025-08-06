#pragma once

// CUDA-specific fix for std::array compatibility
// This header must be included before any standard library headers in CUDA files
#ifdef __CUDACC__
#include <array>
#include <cstddef>
#include <type_traits>
#include <functional>
#endif
