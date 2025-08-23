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
    sys_time<std::chrono::nanoseconds> tp;
    ss >> std::chrono::parse("%Y-%m-%dT%H:%M:%S.%fZ", tp);
    if (ss.fail()) {
        throw std::runtime_error("Invalid timestamp: " + timestamp_str);
    }
    return time_point_cast<system_clock::duration>(tp);
}

} // namespace sep::common
