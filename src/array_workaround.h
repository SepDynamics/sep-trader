#pragma once

// GCC 11 functional header bug workaround
// The functional header uses unqualified 'array' but doesn't include <array>
// This header should be included before any header that might include <functional>

#include <array>

// Make array available in global namespace for GCC 11 functional header
#ifdef __GNUC__
#if __GNUC__ == 11
using std::array;
#endif
#endif
