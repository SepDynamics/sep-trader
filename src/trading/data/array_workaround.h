#pragma once

// Fix for g++ with GCC 11 libstdc++ functional header issue
#ifdef __clang__
#define _GLIBCXX_USE_NOEXCEPT noexcept
#endif

#include <array>
