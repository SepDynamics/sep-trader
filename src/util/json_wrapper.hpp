/**
 * @file json_wrapper.hpp
 * @brief Wrapper for nlohmann/json that fixes std::array inclusion issues
 * 
 * This wrapper ensures that <array> is included before nlohmann/json.hpp
 * to work around a GCC 11+ compilation issue where the functional header
 * uses std::array without including the array header.
 */

#pragma once

// CRITICAL: Include array before nlohmann/json to fix GCC 11+ compilation issues
#include <array>

// Now include the actual json library
#include "nlohmann_json_safe.h"

// Re-export the nlohmann namespace for convenience
namespace sep {
    using json = nlohmann::json;
}