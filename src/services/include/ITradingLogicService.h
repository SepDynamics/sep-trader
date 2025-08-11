#pragma once

#include "IService.h"
#include "TradingTypes.h"
#include "PatternTypes.h"
#include <vector>
#include <string>
#include <map>
#include <functional>

namespace sep {
namespace services {

/**
 * Interface for the Trading Logic Service
 * Responsible for market data processing, signal generation,
 * trading decisions, and performance tracking
 */
class ITradingLogicService : public IService {
public:
    /**
     * Process market data and update internal state
     * @param dataPoint New market data point
     * @return Result<void> Success or error
     */
    virtual Result<void> processMarketData(const MarketDataPoint& dataPoint) = 0;
    
    /**
     * Process a batch of market data points
     * @param dataPoints Collection of market data points
     * @return Result<void> Success or error
     */
    virtual Result<void> processMarketDataBatch(
        const std::vector<MarketDataPoint>& dataPoints) = 0;
    
    /**
     * Update an OHLCV candle with new market data
     * @param symbol Symbol identifier
     * @param timeframe Candle timeframe
     * @param dataPoint New market data point
     * @return Result containing updated candle or error
     */
    virtual Result<OHLCVCandle> updateOHLCVCandle(
        const std::string& symbol,
        TradingTimeframe timeframe,
        const MarketDataPoint& dataPoint) = 0;
    
    /**
     * Get historical OHLCV candles
     * @param symbol Symbol identifier
     * @param timeframe Candle timeframe
     * @param count Number of candles to retrieve
     * @param endTime End time for the last candle (default: now)
     * @return Result containing OHLCV candles or error
     */
    virtual Result<std::vector<OHLCVCandle>> getHistoricalCandles(
        const std::string& symbol,
        TradingTimeframe timeframe,
        int count,
        std::chrono::system_clock::time_point endTime = 
            std::chrono::system_clock::now()) = 0;
    
    /**
     * Generate trading signals based on current market context
     * @param context Current market context
     * @param patternIds Optional patterns to consider
     * @return Result containing generated signals or error
     */
    virtual Result<std::vector<TradingSignal>> generateSignals(
        const MarketContext& context,
        const std::vector<std::string>& patternIds = {}) = 0;
    
    /**
     * Generate trading signals based on recognized patterns
     * @param patterns Patterns to analyze
     * @param context Current market context
     * @return Result containing generated signals or error
     */
    virtual Result<std::vector<TradingSignal>> generateSignalsFromPatterns(
        const std::vector<std::shared_ptr<Pattern>>& patterns,
        const MarketContext& context) = 0;
    
    /**
     * Make trading decisions based on signals
     * @param signals Trading signals
     * @param context Current market context
     * @return Result containing trading decisions or error
     */
    virtual Result<std::vector<TradingDecision>> makeDecisions(
        const std::vector<TradingSignal>& signals,
        const MarketContext& context) = 0;
    
    /**
     * Evaluate performance of past trading decisions
     * @param decisions Past trading decisions
     * @param currentContext Current market context
     * @return Result containing performance metrics or error
     */
    virtual Result<PerformanceMetrics> evaluatePerformance(
        const std::vector<TradingDecision>& decisions,
        const MarketContext& currentContext) = 0;
    
    /**
     * Backtest a trading strategy
     * @param historicalData Historical market data
     * @param parameters Strategy parameters
     * @return Result containing performance metrics or error
     */
    virtual Result<PerformanceMetrics> backtestStrategy(
        const std::map<std::string, std::vector<OHLCVCandle>>& historicalData,
        const std::map<std::string, double>& parameters) = 0;
    
    /**
     * Register a callback for trading signals
     * @param callback Function to call when new signals are generated
     * @return Subscription ID for the callback
     */
    virtual int registerSignalCallback(
        std::function<void(const TradingSignal&)> callback) = 0;
    
    /**
     * Unregister a signal callback
     * @param subscriptionId ID returned from registerSignalCallback
     * @return Result<void> Success or error
     */
    virtual Result<void> unregisterSignalCallback(int subscriptionId) = 0;
    
    /**
     * Get list of available trading strategies
     * @return Map of strategy names to descriptions
     */
    virtual std::map<std::string, std::string> getAvailableStrategies() const = 0;
    
    /**
     * Get current market context
     * @return Result containing current market context or error
     */
    virtual Result<MarketContext> getCurrentMarketContext() const = 0;
};

} // namespace services
} // namespace sep