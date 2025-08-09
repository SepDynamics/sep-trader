#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <execution>
#include <future>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cufft.h>
#endif
