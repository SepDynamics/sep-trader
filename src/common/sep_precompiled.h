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

// CRITICAL: Clean up any macro pollution that might corrupt std::array
// This must be done AFTER array is included but BEFORE other headers
#undef array

// Standard C++ headers that are commonly used
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <queue>
#include <stack>
#include <deque>
#include <list>
#include <optional>
#include <variant>
#include <type_traits>
#include <utility>
#include <tuple>
#include <regex>
#include <random>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <cassert>
#include <limits>
#include <numeric>

// System headers
#include <unistd.h>
#include <sys/types.h>

#endif // SEP_PRECOMPILED_H
