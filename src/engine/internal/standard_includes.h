#pragma once

// 1. CRITICAL: Include the REAL <array> header FIRST, before anything else.
#include <array>

// 2. NOW include all your other essential standard library and third-party headers.
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
#include <nlohmann/json.hpp>

// 3. AT THE VERY END, forcefully undefine the conflicting macro.
// This cleans up any pollution from headers included above, ensuring that
// when the compiler proceeds to the actual .cpp/.cu source files,
// 'array' is no longer a macro.
#undef array
