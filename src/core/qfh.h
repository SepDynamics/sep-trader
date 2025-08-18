#pragma once

#include <cstdint>
#include "core/forward_window_result.h"

namespace sep {
namespace quantum {
namespace bitspace {

// ForwardWindowResult is now defined in forward_window_result.h

/**
 * QFH (Quantum Field Harmonics) configuration constants
 */
namespace qfh {
    constexpr double DEFAULT_LAMBDA = 0.1;  // Default decay constant
    constexpr int MAX_PACKAGE_SIZE = 1024;  // Maximum bit package size
    constexpr int MIN_PACKAGE_SIZE = 8;     // Minimum bit package size
}

} // namespace bitspace
} // namespace quantum
} // namespace sep