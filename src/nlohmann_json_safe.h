#pragma once

// Safe nlohmann/json wrapper that ensures std::array is available first
#ifndef NLOHMANN_JSON_SAFE_INCLUDED
#define NLOHMANN_JSON_SAFE_INCLUDED

// Force std::array to be available
#include <array>

// Pre-include all std headers that nlohmann/json needs
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <functional>
#include <tuple>
#include <initializer_list>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <sstream>

// Ensure array is in std namespace
using std::array;

// Now include nlohmann/json with all dependencies available
#include <nlohmann/json.hpp>

#endif // NLOHMANN_JSON_SAFE_INCLUDED
