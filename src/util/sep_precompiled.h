/**
 * @file sep_precompiled.h
 * @brief A simplified, stable pre-compiled header for the SEP project.
 *
 * This header includes only the most common and non-problematic C++ headers.
 * All complex compiler-specific fixes have been moved to dedicated headers.
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
// Project-Specific Headers
//==============================================================================
// nlohmann_json wrapper - this now contains its own fixes
#include "nlohmann_json_safe.h"