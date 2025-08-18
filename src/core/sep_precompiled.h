/**
 * @file sep_precompiled.h  
 * @brief Consolidated precompiled header for SEP Professional Trader-Bot
 * 
 * This is the canonical PCH used by the CMake build system. It includes
 * essential C++ standard library headers and SEP project-specific headers
 * in dependency order for optimal compilation performance.
 */

#pragma once

//==============================================================================
// C++ Standard Library Headers
//==============================================================================
// Core C++ headers
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <algorithm>
#include <functional>
#include <csignal>

// Containers
#include <map>
#include <unordered_map>
#include <set>

// Utilities
#include <optional>
#include <stdexcept>

// Filesystem
#include <filesystem>

//==============================================================================
// Third-Party Libraries
//==============================================================================
#include <fmt/format.h>
#include <spdlog/spdlog.h>

//==============================================================================
// SEP Foundation Headers
//==============================================================================
// Core stability and compatibility layers
#include "util/stable_headers.h"
#include "cuda_compatibility.h"

// Safe third-party wrappers
#include "util/nlohmann_json_safe.h"

//==============================================================================
// SEP Core Data Types
//==============================================================================
#include "util/result.h"
#include "util/financial_data_types.h"

//==============================================================================
// SEP I/O and Connectivity
//==============================================================================
#include "io/oanda_connector.h"

//==============================================================================
// SEP Pattern Processing
//==============================================================================
#include "pattern_types.h"
#include "forward_window_result.h"
#include "trajectory.h"
#include "trace.hpp"

//==============================================================================
// SEP CUDA Infrastructure
//==============================================================================
#include "stream.h"
#include "kernels.h"
#include "cuda/cuda_error.cuh"

//==============================================================================
// SEP Utilities
//==============================================================================
#include "util/common.h"
#include "util/logging.h"
#include "util/types.h"
#include "util/qfh.h"
#include "util/placeholder_detection.h"

//==============================================================================
// SEP Memory Management
//==============================================================================
#include "util/memory_tier_manager.hpp"

//==============================================================================
// SEP Global Configuration
//==============================================================================
#include "global_includes.h"
#include "ticker_optimization_kernel.cuh"