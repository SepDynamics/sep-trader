#pragma once

// Safe nlohmann/json wrapper that ensures std::array is available first
#ifndef NLOHMANN_JSON_SAFE_INCLUDED
#define NLOHMANN_JSON_SAFE_INCLUDED

// Pre-include all std headers that nlohmann/json needs
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

// Now include nlohmann/json with all dependencies available
#include <nlohmann/json.hpp>

#endif // NLOHMANN_JSON_SAFE_INCLUDED
