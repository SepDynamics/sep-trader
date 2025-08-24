#pragma once
#include <string_view>
#include <iostream>

namespace sep {
namespace trace {

#ifdef SEP_ENABLE_TRACE
inline void log(std::string_view stage, std::string_view detail) {
    std::clog << "[TRACE] " << stage << ": " << detail << '\n';
}
#else
inline void log(std::string_view, std::string_view) noexcept {}
#endif

} // namespace trace
} // namespace sep
