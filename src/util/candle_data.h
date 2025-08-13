#pragma once

#include <cstdint>
#include <string>

namespace sep::common {

/**
 * Structure representing OHLC (Open, High, Low, Close) candle data for financial markets
 */
struct CandleData {
    double open;
    double high;
    double low;
    double close;
    double volume;
    uint64_t timestamp_ns; // nanoseconds since epoch
    
    // Default constructor
    CandleData() : open(0), high(0), low(0), close(0), volume(0), timestamp_ns(0) {}
    
    // Full constructor
    CandleData(double o, double h, double l, double c, double v, uint64_t ts)
        : open(o), high(h), low(l), close(c), volume(v), timestamp_ns(ts) {}
};

/**
 * Enum representing different candle timeframes
 */
enum class CandleTimeframe {
    M1,   // 1 minute
    M5,   // 5 minutes
    M15,  // 15 minutes
    M30,  // 30 minutes
    H1,   // 1 hour
    H4,   // 4 hours
    D1,   // 1 day
    W1,   // 1 week
    MN1   // 1 month
};

/**
 * Convert timeframe to seconds
 */
inline int64_t timeframeToSeconds(CandleTimeframe tf) {
    switch (tf) {
        case CandleTimeframe::M1:  return 60;
        case CandleTimeframe::M5:  return 300;
        case CandleTimeframe::M15: return 900;
        case CandleTimeframe::M30: return 1800;
        case CandleTimeframe::H1:  return 3600;
        case CandleTimeframe::H4:  return 14400;
        case CandleTimeframe::D1:  return 86400;
        case CandleTimeframe::W1:  return 604800;
        case CandleTimeframe::MN1: return 2592000; // 30 days approximation
        default: return 0;
    }
}

} // namespace sep::common