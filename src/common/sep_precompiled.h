/**
 * @file sep_precompiled.h
 * @brief Primary pre-compiled header for SEP project
 *
 * This header includes all common dependencies and provides
 * critical fixes for std::array issues that affect nlohmann_json
 * and other third-party libraries under CUDA compilation.
 */

#pragma once

// CRITICAL COMPILER WORKAROUND:
// GCC 11 functional header has a bug where it uses unqualified 'array' 
// within std namespace but doesn't include <array> itself.
// We MUST include array before any header that might pull in functional.

#include <array>

// CRITICAL: Clean up any macro pollution that might corrupt std::array
// This must be done AFTER array is included but BEFORE other headers

// Standard library includes
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <condition_variable>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <execution>
#include <future>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <deque>
#include <list>
#include <variant>
#include <tuple>
#include <regex>
#include <random>
#include <cstring>
#include <cassert>
#include <limits>

// Third-party library includes (exclude GLM from CUDA compilation to avoid conflicts)
#ifndef __CUDACC__
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#endif

#include <fmt/format.h>
#include <spdlog/spdlog.h>

// System headers
#include <unistd.h>
#include <sys/types.h>


#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cufft.h>
#endif
