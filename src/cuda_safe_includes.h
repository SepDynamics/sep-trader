#pragma once

// CUDA-safe includes that avoid std::array conflicts
// Always include array first before anything else that might need it
#include <array>
#include <queue>
#include <functional>
#include <type_traits>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>

// For CUDA compilation, ensure we have proper array support
#ifdef __CUDACC__
// Verify std::array is available 
static_assert(std::is_same_v<std::array<int, 1>, std::array<int, 1>>, "std::array must be available for CUDA");
#endif
