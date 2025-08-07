#pragma once

// CRITICAL: Include comprehensive array protection FIRST
#include "../../array_protection.h"

// 4. NOW include all other essential standard library and third-party headers.
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

// 5. AT THE VERY END, forcefully undefine any conflicting macro again.
// This ensures that when the compiler proceeds to actual .cpp/.cu source files,
// 'array' is no longer a macro and std::array can be used correctly.
#ifdef array
#undef array
#endif
