#ifndef NLOHMANN_JSON_SAFE_INCLUDED
#define NLOHMANN_JSON_SAFE_INCLUDED

//==============================================================================
// Self-contained header to ensure nlohmann_json compiles correctly.
//==============================================================================

// 1. C Standard Library Headers
// These are required for basic types and functions used by the C++ STL.
#include <cstddef>
#include <cstdint>
#include <cstring>

// 2. CRITICAL: Include <array> before anything else
// This is the primary fix for the GCC 11 std::array bug.
#include <array>

// 3. Other required C++ Standard Library Headers
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

// 4. Include the nlohmann_json library
#include <nlohmann/json.hpp>

#endif // NLOHMANN_JSON_SAFE_INCLUDED