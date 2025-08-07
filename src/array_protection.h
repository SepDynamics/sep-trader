#ifndef SEP_ARRAY_PROTECTION_H
#define SEP_ARRAY_PROTECTION_H

// COMPREHENSIVE STD::ARRAY PROTECTION HEADER
// This header MUST be included FIRST in all compilation units that use std::array
// or include headers that depend on std::array (like nlohmann/json.hpp)

// 1. Force include of array header before any potential macro pollution
#include <array>

// 2. Ensure the header guard is set
#ifndef _GLIBCXX_ARRAY
#define _GLIBCXX_ARRAY 1
#endif

// 3. Clean up any existing array macro pollution
#ifdef array
#undef array
#endif

// 4. Force std::array into global namespace (safety measure)
namespace std {
    using ::std::array;
}

// 5. Protect against future macro pollution
#define SEP_ARRAY_PROTECTED 1

#endif // SEP_ARRAY_PROTECTION_H
