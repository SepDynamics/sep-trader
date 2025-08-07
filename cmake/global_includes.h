#pragma once

// This file MUST be included first to prevent macro conflicts with std::array

// Step 1: Include array before anything else can pollute it
#include <array>

// Step 2: Immediately save std::array in case it gets corrupted later
namespace __array_guard {
    template<typename T, std::size_t N>
    using safe_array = std::array<T, N>;
}

// Step 3: Include other essential headers that should be available globally
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <optional>
#include <utility>
#include <cstddef>
#include <cstdint>

// Step 4: Force undefine any array macros and restore
#ifdef array
#undef array
#endif

// Step 5: TBB headers (after array protection)
#ifdef __has_include
  #if __has_include(<oneapi/tbb.h>)
    #include <oneapi/tbb.h>
  #elif __has_include(<tbb/tbb.h>)
    #include <tbb/tbb.h>
  #elif __has_include(<tbb/task.h>)
    #include <tbb/task.h>
  #endif
#else
  #include <tbb/task.h>
#endif

// Step 6: Final cleanup - aggressively undefine any conflicting macros
#ifdef array
#undef array
#endif

// Step 7: Restore std::array if it was corrupted (fallback mechanism)
#ifndef std
namespace std {
    template<typename T, std::size_t N>
    using array = ::__array_guard::safe_array<T, N>;
}
#endif
