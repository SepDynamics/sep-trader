#pragma once

#include "../common/ServiceBase.h"
#include "../include/ITradingLogicService.h"
#include "../include/Result.h"
#include <map>
#include <mutex>
#include <vector>
#include <atomic>
#include <functional>
#include <unordered_map>

namespace sep {
namespace services {

/**
 * Implementation of the Trading Logic Service
 * Responsible for market data processing, signal generation,
 * trading decisions, and performance tracking
 */
class TradingLogicService : public ITradingLogicService, public ServiceBase {
public:
    TradingLogicService();
    ~TradingLogicService() override;

    // IService interface implementation - override from ServiceBase to resolve diamond inheritance
    bool isReady() const override;

    // ITradingLogicService interface implementation
    Result<void> processMarketData(const MarketDataPoint& dataPoint) override;
    Result<void> processMarketDataBatch(const std::vector<MarketDataPoint>& dataPoints) override;
    Result<OHLCVCandle> updateOHLCVCandle(const std::string& symbol, TradingTimeframe timeframe, 
                                          const MarketDataPoint& dataPoint) override;
    Result<std::vector<OHLCVCandle>> getHistoricalCandles(const std::string& symbol, 
                                                          TradingTimeframe timeframe, 
                                                          int count,
                                                          std::chrono::system_clock::time_point endTime) override;
    Result<std::vector<TradingSignal>> generateSignals(const MarketContext& context, 
                                                      const std::vector<std::string>& patternIds) override;
    Result<std::vector<TradingSignal>> generateSignalsFromPatterns(
        const std::vector<std::shared_ptr<Pattern>>& patterns, 
        const MarketContext& context) override;
    Result<std::vector<TradingDecision>> makeDecisions(const std::vector<TradingSignal>& signals, 
                                                      const MarketContext& context) override;
    Result<PerformanceMetrics> evaluatePerformance(const std::vector<TradingDecision>& decisions, 
                                                  const MarketContext& currentContext) override;
    Result<PerformanceMetrics> backtestStrategy(
        const std::map<std::string, std::vector<OHLCVCandle>>& historicalData,
        const std::map<std::string, double>& parameters) override;
    int registerSignalCallback(std::function<void(const TradingSignal&)> callback) override;
    Result<void> unregisterSignalCallback(int subscriptionId) override;
    std::map<std::string, std::string> getAvailableStrategies() const override;
    Result<MarketContext> getCurrentMarketContext() const override;

protected:
    // ServiceBase required overrides
    Result<void> onInitialize() override;
    Result<void> onShutdown() override;

private:
    // Helper methods
    std::string generateUniqueId(const std::string& prefix) const;
    void notifySignalCallbacks(const TradingSignal& signal);
    double calculateConfidence(const MarketContext& context, const TradingActionType& action) const;
    
    // Internal data structures
    std::map<std::string, std::map<TradingTimeframe, std::vector<OHLCVCandle>>> candleData_;
    std::map<std::string, MarketDataPoint> latestMarketData_;
    std::vector<TradingDecision> recentDecisions_;
    MarketContext currentContext_;
    
    // Strategies
    std::map<std::string, std::string> availableStrategies_;
    std::map<std::string, std::function<std::vector<TradingSignal>(const MarketContext&)>> strategyFunctions_;
    
    // Signal callbacks
    std::unordered_map<int, std::function<void(const TradingSignal&)>> signalCallbacks_;
    std::atomic<int> nextCallbackId_;
    
    // Thread safety
    mutable std::mutex dataMutex_;
    mutable std::mutex callbackMutex_;
};

} // namespace services
} // namespace sep