#pragma once

#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <memory>
#include <chrono>

namespace sep {
namespace services {

/**
 * Trading action types
 */
enum class TradingActionType {
    Buy,
    Sell,
    Hold,
    Hedge,
    Rebalance
};

/**
 * Trading timeframe
 */
enum class TradingTimeframe {
    Tick,       // Individual tick
    Second,     // 1 second
    Minute,     // 1 minute
    Minute5,    // 5 minutes
    Minute15,   // 15 minutes
    Minute30,   // 30 minutes
    Hour,       // 1 hour
    Hour4,      // 4 hours
    Day,        // 1 day
    Week,       // 1 week
    Month       // 1 month
};

/**
 * Market data point
 */
struct MarketDataPoint {
    std::string symbol;
    double price;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    std::map<std::string, double> additionalFields;
    
    MarketDataPoint() : price(0.0), volume(0.0) {}
};

/**
 * OHLCV (Open, High, Low, Close, Volume) candle
 */
struct OHLCVCandle {
    std::string symbol;
    TradingTimeframe timeframe;
    double open;
    double high;
    double low;
    double close;
    double volume;
    std::chrono::system_clock::time_point openTime;
    std::chrono::system_clock::time_point closeTime;
    
    OHLCVCandle() : 
        open(0.0), high(0.0), low(0.0), close(0.0), volume(0.0),
        timeframe(TradingTimeframe::Minute) {}
};

/**
 * Trading signal
 */
struct TradingSignal {
    std::string signalId;
    std::string symbol;
    TradingActionType actionType;
    double confidence;
    double targetPrice;
    double stopLoss;
    std::chrono::system_clock::time_point generatedTime;
    std::chrono::system_clock::time_point expirationTime;
    std::map<std::string, double> indicators;
    std::string patternId; // Related pattern ID if applicable
    
    TradingSignal() : 
        actionType(TradingActionType::Hold),
        confidence(0.0),
        targetPrice(0.0),
        stopLoss(0.0) {}
};

/**
 * Trading decision
 */
struct TradingDecision {
    std::string decisionId;
    std::string symbol;
    TradingActionType action;
    double quantity;
    double price;
    double confidence;
    std::vector<std::string> signalIds; // Related signal IDs
    std::chrono::system_clock::time_point timestamp;
    std::string reason;
    
    TradingDecision() : 
        action(TradingActionType::Hold),
        quantity(0.0),
        price(0.0),
        confidence(0.0) {}
};

/**
 * Performance metrics
 */
struct PerformanceMetrics {
    double totalReturn;
    double sharpeRatio;
    double maxDrawdown;
    double winRate;
    double profitFactor;
    int totalTrades;
    std::map<std::string, double> additionalMetrics;
    
    PerformanceMetrics() : 
        totalReturn(0.0),
        sharpeRatio(0.0),
        maxDrawdown(0.0),
        winRate(0.0),
        profitFactor(0.0),
        totalTrades(0) {}
};

/**
 * Market context
 */
struct MarketContext {
    std::map<std::string, std::vector<OHLCVCandle>> historicalData;
    std::map<std::string, MarketDataPoint> currentPrices;
    std::map<std::string, double> indicators;
    std::map<std::string, double> marketMetrics;
    
    MarketContext() {}
};

} // namespace services
} // namespace sep