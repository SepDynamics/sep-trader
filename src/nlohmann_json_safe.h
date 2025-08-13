#pragma once

// Include required standard headers first
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <initializer_list>
#include <memory>

// Disable exceptions for performance and to avoid issues with CUDA
#define NLOHMANN_JSON_NOEXCEPTION 1

// Include the nlohmann/json header
#include <nlohmann/json.hpp>

// Export commonly used types
namespace sep {
namespace json {
    using json = nlohmann::json;
    using json_ref = nlohmann::json&;
    using const_json_ref = const nlohmann::json&;
} // namespace json
} // namespace sep
