#pragma once

#include <array>
#include <chrono>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "connectors/oanda_connector.h"
#include "engine/internal/standard_includes.h"
#include <nlohmann/json.hpp>

namespace sep::trading {

// Configuration for unified data management
struct UnifiedDataConfig {
    // Remote sync settings (from RemoteDataManager)
    std::string remote_host = "localhost";
    int remote_port = 5432;
    std::string db_name = "sep_trading";
    std::string redis_host = "localhost";
    int redis_port = 6379;
    std::string data_path = "/opt/sep-data";
    
    // Local cache settings (from DataCacheManager)  
    std::string local_cache_path = ".cache/trading_data";
    int cache_ttl_hours = 48;  // Extended for 48H cache
    bool enable_compression = true;
    
    // Live trading cache settings
    std::string live_cache_path = ".cache/live_data";
    int live_cache_ttl_hours = 2;  // Short TTL for live data
};

// Training data structure (from RemoteDataManager)
struct TrainingData {
    std::string pair;
    std::chrono::system_clock::time_point timestamp;
    std::vector<double> features;
    double target;
    std::string metadata;
};

// Model state structure (from RemoteDataManager)  
struct ModelState {
    std::string model_id;
    std::string pair;
    std::vector<uint8_t> weights;
    double accuracy;
    std::chrono::system_clock::time_point trained_at;
    nlohmann::json hyperparameters;
};

// Cache info structure (from DataCacheManager)
struct CacheInfo {
    std::chrono::system_clock::time_point last_update;
    size_t data_points;
    std::string cache_file;
    bool valid;
};

/**
 * Unified Trading Data Manager
 * Consolidates RemoteDataManager + DataCacheManager functionality
 * Supports both training data sync and live trading cache
 */
class UnifiedDataManager {
public:
    explicit UnifiedDataManager(const UnifiedDataConfig& config);
    ~UnifiedDataManager();

    // ============ LIVE TRADING CACHE API (from DataCacheManager) ============
    
    /**
     * Initialize with OANDA connector for live trading
     */
    bool initializeLiveTrading(sep::connectors::OandaConnector* connector);
    
    /**
     * Legacy API compatibility - same as initializeLiveTrading
     */
    bool initialize(sep::connectors::OandaConnector* connector) {
        return initializeLiveTrading(connector);
    }
    
    /**
     * Ensure 48H cache is available and fresh for live trading
     */
    bool ensureLiveCacheReady(const std::string& instrument = "EUR_USD");
    
    /**
     * Get cached live trading data (48H window)
     */
    std::vector<sep::connectors::MarketData> getLiveCachedData(const std::string& instrument = "EUR_USD");
    
    /**
     * Force refresh of live cache
     */
    bool refreshLiveCache(const std::string& instrument = "EUR_USD");
    
    /**
     * Get cache status for live trading
     */
    CacheInfo getLiveCacheInfo(const std::string& instrument = "EUR_USD");

    // ============ TRAINING DATA SYNC API (from RemoteDataManager) ============
    
    /**
     * Fetch training data from remote
     */
    std::future<std::vector<TrainingData>> fetchTrainingDataAsync(
        const std::string& pair, 
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    );
    
    /**
     * Push trained model to remote
     */
    std::future<bool> pushModelAsync(const ModelState& model);
    
    /**
     * Pull latest model from remote
     */
    std::future<std::optional<ModelState>> pullLatestModelAsync(const std::string& pair);
    
    /**
     * Sync local cache with remote
     */
    std::future<bool> syncToRemoteAsync();
    
    /**
     * Check if remote is available
     */
    bool isRemoteAvailable();

    // ============ UNIFIED API ============
    
    /**
     * Get unified cache status across both systems
     */
    struct UnifiedCacheStatus {
        CacheInfo live_cache;
        bool remote_available;
        size_t training_data_count;
        size_t model_count;
    };
    
    UnifiedCacheStatus getUnifiedStatus();
    
    /**
     * Cleanup old cache files
     */
    void cleanupCache();

private:
    UnifiedDataConfig config_;
    sep::connectors::OandaConnector* oanda_connector_ = nullptr;
    
    // Cache management
    std::mutex cache_mutex_;
    std::string getLiveCacheFile(const std::string& instrument);
    std::string getTrainingCacheFile(const std::string& pair);
    
    // Remote connection management
    std::mutex remote_mutex_;
    bool remote_available_ = false;
    
    // Internal helpers
    bool validateCacheFile(const std::string& filepath, int ttl_hours);
    bool saveCacheFile(const std::string& filepath, const std::vector<sep::connectors::MarketData>& data);
    std::vector<sep::connectors::MarketData> loadCacheFile(const std::string& filepath);
};

} // namespace sep::trading
