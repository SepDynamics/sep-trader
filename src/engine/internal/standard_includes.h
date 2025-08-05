#pragma once

// GCC 11 functional header bug workaround - must come first
#include <array>
#ifdef __GNUC__
#if __GNUC__ == 11
using std::array;
#endif
#endif

#include <vector>
#include <string>
#include <memory>
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
#include <functional>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
