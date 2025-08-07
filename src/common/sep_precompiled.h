// SEP Professional Pre-compiled Header
// This MUST be included first in ALL source files to prevent header conflicts

#ifndef SEP_PRECOMPILED_H
#define SEP_PRECOMPILED_H

// CRITICAL COMPILER WORKAROUND:
// GCC 11 functional header has a bug where it uses unqualified 'array' 
// within std namespace but doesn't include <array> itself.
// We MUST include array before any header that might pull in functional.

#include <array>

// Force the array header guard to be set (following pattern from array_fix.h)
#ifndef _GLIBCXX_ARRAY
#define _GLIBCXX_ARRAY 1
#endif

#endif // SEP_PRECOMPILED_H
