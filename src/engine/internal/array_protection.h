#pragma once

// This header provides a robust solution to the std::array macro conflict
// by ensuring that <array> is included and any conflicting macros are
// undefined before they can cause compilation errors.

#ifdef array
    #undef array
#endif

#include <array>
