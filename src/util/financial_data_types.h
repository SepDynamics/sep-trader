#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "core/candle_data.h"

namespace sep::common {

enum class MultiTimeframeSignal {
    STRONG_BUY,
    BUY,
    NEUTRAL,
    SELL,
    STRONG_SELL
};

struct SEPSignalData {
    std::string signal_id;
    std::chrono::time_point<std::chrono::system_clock> timestamp;
    double signal_value;
    double coherence;
    double stability;
    double entropy;
    double alpha_signal;
    double trend_strength;
    MultiTimeframeSignal signal_type;
};

struct CorrelationMetrics {
    double coherence_pearson;
    double coherence_spearman;
    double stability_pearson;
    double stability_spearman;
    double entropy_pearson;
    double entropy_spearman;
    int sample_count;
};


enum class OrderStatus { PENDING, FILLED, CANCELED };

struct OrderInfo {
    std::string id;
    std::string instrument;
    double units{0};
    double price{0};
    OrderStatus status{OrderStatus::PENDING};
};

// Parse an ISO 8601 timestamp in UTC (e.g. "2024-07-20T12:34:56.789000Z")
// into a std::chrono::system_clock::time_point. Fractional seconds up to
// nanosecond precision are supported. Throws std::runtime_error on invalid
// input.
std::chrono::time_point<std::chrono::system_clock> parseTimestamp(const std::string& timestamp_str);

// Convert a time_point to nanoseconds since epoch
int64_t time_point_to_nanoseconds(const std::chrono::time_point<std::chrono::system_clock>& tp);

} // namespace sep::common
