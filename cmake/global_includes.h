#pragma once

// CRITICAL: Force std::array definition before any other headers that might conflict
#ifndef _GLIBCXX_ARRAY
#include <array>
#endif

// Core C++ headers that need to be available globally
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

// Note: nlohmann/json.hpp removed from global includes to avoid header pollution
// Include it directly in files that need it

// TBB headers - try modern path first
#ifdef __has_include
  #if __has_include(<oneapi/tbb.h>)
    #include <oneapi/tbb.h>
  #elif __has_include(<tbb/tbb.h>)
    #include <tbb/tbb.h>
  #elif __has_include(<tbb/task.h>)
    #include <tbb/task.h>
  #endif
#else
  // Fallback for older compilers
  #include <tbb/task.h>
#endif

// CRITICAL: Undefine any conflicting 'array' macro that might be interfering with std::array
// This must come AFTER all system headers that might define it
#ifdef array
#undef array
#endif