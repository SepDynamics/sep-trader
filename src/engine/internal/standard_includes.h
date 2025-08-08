#pragma once

// CRITICAL: Include nlohmann_json_protected.h FIRST to ensure proper array setup
#include "../../nlohmann_json_protected.h"

// Include array early and often to prevent conflicts
#include <array>

// Now include standard library headers in safe order
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <algorithm>
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

// 5. AT THE VERY END, forcefully undefine any conflicting macro again.
// This ensures that when the compiler proceeds to actual .cpp/.cu source files,
// 'array' is no longer a macro and std::array can be used correctly.
#ifdef array
#undef array
#endif
