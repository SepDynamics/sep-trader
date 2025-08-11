#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace sep::testbed {

inline bool strict_placeholder_check_enabled() {
    const char* env = std::getenv("SEP_STRICT_PLACEHOLDER_CHECK");
    return env && std::string_view(env) == "1";
}

inline void ensure_not_placeholder(std::string_view value, std::string_view placeholder = "PLACEHOLDER") {
    if (strict_placeholder_check_enabled() && value == placeholder) {
        throw std::runtime_error("Placeholder value detected in production path");
    }
}

} // namespace sep::testbed

