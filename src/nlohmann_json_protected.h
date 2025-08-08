#ifndef SEP_NLOHMANN_JSON_PROTECTED_H
#define SEP_NLOHMANN_JSON_PROTECTED_H

// COMPREHENSIVE NLOHMANN JSON PROTECTION HEADER
// This header MUST be used instead of direct #include <nlohmann/json.hpp>
// to prevent std::array macro conflicts

// 1. Include our comprehensive array protection FIRST
#include "array_protection.h"

// 2. Force include array header before nlohmann
#include <array>

// 3. Clean up any remaining array macro pollution
#ifdef array
#undef array
#endif

// 4. Now safely include nlohmann json
#include <nlohmann/json.hpp>

// 5. Final cleanup - ensure array macro is gone
#ifdef array
#undef array
#endif

// 6. Ensure std::array is properly available
using std::array;

#endif // SEP_NLOHMANN_JSON_PROTECTED_H
