#pragma once

#include <ctime>
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>

// Candle structure for multi-timeframe analysis
struct Candle {
    std::uint64_t timestamp;  // epoch seconds
    double open;
    double high;
    double low;
    double close;
    double volume;
};

// Helper function to parse ISO timestamp to epoch seconds
inline std::uint64_t parseTimestamp(const std::string& time_str) {
    std::tm tm = {};
    std::istringstream ss(time_str);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return static_cast<std::uint64_t>(std::mktime(&tm));
}
