/**
 * @file sep_precompiled.h  
 * @brief Consolidated precompiled header for SEP Professional Trader-Bot
 * 
 * This is the canonical PCH used by the CMake build system. It includes
 * essential C++ standard library headers and SEP project-specific headers
 * in dependency order for optimal compilation performance.
 */

#pragma once

// C++20 compatibility for C builds (must be first)
#include "common/namespace_protection.hpp"

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
#include "core/cuda_compatibility.h"
#include "util/stable_headers.h"

// NOTE: nlohmann_json_safe.h removed - already included in stable_headers.h

//==============================================================================
// SEP Core Data Types
//==============================================================================
#include "core/standard_includes.h"
#include "core/result_types.h"
#include "core/types.h"
#include "util/financial_data_types.h"

//==============================================================================
// SEP I/O and Connectivity
//==============================================================================
#include "io/oanda_connector.h"

//==============================================================================
// SEP Pattern Processing
//==============================================================================
#include "pattern_types.h"
#include "core/forward_window_result.h"
#include "trajectory.h"
#include "core/trace.hpp"

//==============================================================================
// SEP CUDA Infrastructure
//==============================================================================
#include "cuda/stream.h"
#include "kernels.h"
#include "core/cuda_error.cuh"

//==============================================================================
// SEP Utilities
//==============================================================================
#include "core/common.h"
#include "core/logging.h"
#include "core/qfh.h"

//==============================================================================
// SEP Memory Management
//==============================================================================
#include "util/memory_tier_manager.hpp"

//==============================================================================
// SEP Global Configuration
//==============================================================================
#include "util/global_includes.h"
#include "cuda/ticker_optimization_kernel.cuh"