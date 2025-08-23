#include "financial_data_types.h"
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <chrono>

namespace sep::common {

int64_t time_point_to_nanoseconds(const std::chrono::time_point<std::chrono::system_clock>& tp) {
    auto duration = tp.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

std::chrono::time_point<std::chrono::system_clock> parseTimestamp(const std::string& timestamp_str) {
    using namespace std::chrono;

    std::istringstream ss(timestamp_str);
    std::tm tm{};
    char dot;
    long fractional = 0;

    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        throw std::runtime_error("Invalid timestamp: " + timestamp_str);
    }

    if (ss.peek() == '.') {
        ss >> dot >> fractional;
    }
    // Consume trailing 'Z' if present
    if (ss.peek() == 'Z') {
        ss.get();
    }

    auto time = std::mktime(&tm);
    auto tp = system_clock::from_time_t(time);
    tp += std::chrono::nanoseconds(fractional);
    return tp;
}

} // namespace sep::common
