#pragma once
#include <vector>
#include <string>
#include <memory>
#include <map>
#include "io/oanda_connector.h"
#include "app/candle_types.h"
#include "app/quantum_signal_bridge.hpp"
#include "core/signal_types.h"
#include <hiredis/hiredis.h>

namespace sep::apps {

class MarketModelCache {
public:
    explicit MarketModelCache(std::shared_ptr<sep::connectors::OandaConnector> connector,
                              std::shared_ptr<IQuantumPipeline> pipeline = nullptr);

    // Main function: Ensures cache for the last week is ready
    bool ensureCacheForLastWeek(const std::string& instrument = "EUR_USD");
    
    // Accessor for the processed signals
    const std::map<std::string, sep::trading::QuantumTradingSignal>& getSignalMap() const;

    // Process a batch of candles into the cache using the quantum pipeline
    void processBatch(const std::string& instrument, const std::vector<Candle>& candles);

private:
    bool loadCacheFromValkey(const std::string& cache_key);
    bool saveCacheToValkey(const std::string& cache_key) const;
    void processAndCacheData(const std::vector<Candle>& raw_candles, const std::string& cache_key);
    std::string getCacheKeyForLastWeek(const std::string& instrument) const;
    
    // Valkey connection management
    bool connectToValkey();
    void disconnectFromValkey();

    std::shared_ptr<sep::connectors::OandaConnector> oanda_connector_;
    std::map<std::string, sep::trading::QuantumTradingSignal> processed_signals_;
    std::shared_ptr<IQuantumPipeline> pipeline_;
    
    // Valkey connection
    redisContext* valkey_context_;
    std::string valkey_url_;
};

} // namespace sep::apps
