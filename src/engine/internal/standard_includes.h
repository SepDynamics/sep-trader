#pragma once

// Standard includes for SEP engine  
// Include fundamental headers in correct order for g++/GCC compatibility

// Force array inclusion before anything else to prevent GCC compatibility issues
#ifndef _GLIBCXX_ARRAY
#include <array>
#endif

#include <cstddef>
#include <type_traits>

// Ensure functional gets array properly  
#ifndef _GLIBCXX_FUNCTIONAL
#include <array>  // Double-ensure array is available
#include <functional>
#endif

#include <vector>
#include <string>
#include <memory>
#include <algorithm>  // Now safe since <functional> is already included
#include <cmath>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
