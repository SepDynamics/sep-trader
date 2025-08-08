#pragma once

// This file MUST be included first to prevent macro conflicts

// Step 1: Include the array protection header to handle std::array conflicts.
#include "engine/internal/array_protection.h"

// Step 2: Include GLM configuration for consistent vector/matrix types.
#include "engine/internal/glm_config.h"

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

// Step 4: TBB headers
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
