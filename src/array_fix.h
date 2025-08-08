// CRITICAL ARRAY FIX FOR GCC 11 DOCKER BUILD
// This file is force-included via CMake -include flag

#ifndef SEP_ARRAY_FIX_H
#define SEP_ARRAY_FIX_H

// Clean up any macro pollution first
#ifdef array
#undef array
#endif

// Include array header immediately and unconditionally
#include <array>

#endif // SEP_ARRAY_FIX_H
