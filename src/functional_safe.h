#pragma once

// Safe wrapper for functional header to prevent std::array issues
// This ensures array is included before functional to satisfy GCC 11 STL dependencies

#include <array>
#include <tuple>

// Define array as std::array for GCC 11 functional header compatibility


#include <functional>

#undef array
