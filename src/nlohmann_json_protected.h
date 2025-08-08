#ifndef SEP_NLOHMANN_JSON_PROTECTED_H
#define SEP_NLOHMANN_JSON_PROTECTED_H

// SIMPLIFIED NLOHMANN JSON PROTECTION HEADER
// This header provides nlohmann/json with array header pre-included

// Include array before nlohmann/json to resolve std::array dependencies
#include <array>
#include <nlohmann/json.hpp>

#endif // SEP_NLOHMANN_JSON_PROTECTED_H
