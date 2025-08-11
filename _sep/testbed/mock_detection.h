#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include "src/core_types/candle_data.h"

namespace sep::testbed
{

    inline bool strict_mock_check_enabled()
    {
        const char* env = std::getenv("SEP_STRICT_MOCK_CHECK");
        return env && std::string_view(env) == "1";
    }

    inline void ensure_not_mock(const sep::core::CandleData& candle)
    {
        if (strict_mock_check_enabled() && candle.is_mock)
        {
            throw std::runtime_error("Mock candle data detected in production path");
        }
    }

}  // namespace sep::testbed
