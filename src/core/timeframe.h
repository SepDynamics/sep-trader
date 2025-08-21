#pragma once

#include <cstdint>

namespace sep {

enum class Timeframe : uint8_t {
    M1,
    M5,
    M15,
    M30,
    H1,
    H4,
    D1
};

} // namespace sep

