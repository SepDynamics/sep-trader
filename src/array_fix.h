// CRITICAL ARRAY FIX FOR GCC 11 DOCKER BUILD
// This file MUST be included before any other header in problematic files
// to resolve std::array issues in the Docker build environment

#ifndef SEP_ARRAY_FIX_H
#define SEP_ARRAY_FIX_H

// Force include array header to prevent "array is not a member of std" errors
#include <array>

// Ensure the array header guard is properly set
#ifndef _GLIBCXX_ARRAY
#define _GLIBCXX_ARRAY 1
#endif

#endif // SEP_ARRAY_FIX_H
