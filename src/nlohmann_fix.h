#pragma once

// Comprehensive fix for nlohmann/json std::array issues
// This header MUST be included before any nlohmann/json headers

// Force proper includes
#include <array>
#include <cstddef>
#include <type_traits>
#include <functional>

// Undefine any problematic array macros
#ifdef array
#undef array
#endif

// Ensure std namespace is fully available
using std::array;

// Workaround for certain compiler issues
#ifndef _GLIBCXX_ARRAY
#define _GLIBCXX_ARRAY 1
#include <array>
#endif

// Don't redefine std::array - it's already available from #include <array>

// Additional includes that nlohmann headers expect
#include <string>
#include <vector>
#include <map>
#include <cstdint>
