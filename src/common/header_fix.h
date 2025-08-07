#pragma once

// Universal header fix for GCC 11+ functional header issues
// Include this file FIRST in any .cpp file that fails with array/functional errors

#include <array>
#include <functional>

// This ensures std::array is visible before any other header tries to use it
