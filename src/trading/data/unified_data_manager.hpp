#pragma once

// Standard library includes
#include <array>
#include <chrono>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

// Include third-party dependencies
#include <nlohmann/json.hpp>

// Forward declarations
namespace sep {
namespace connectors {
class OandaConnector;
struct MarketData;
}
}

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

class UnifiedDataManager {
public:
    explicit UnifiedDataManager(const UnifiedDataConfig& config);
    ~UnifiedDataManager() = default;

    // ============ LIVE TRADING CACHE API (from DataCacheManager) ============
    
    bool initializeLiveTrading(sep::connectors::OandaConnector* connector);
    
    bool initialize(sep::connectors::OandaConnector* connector) {
        return initializeLiveTrading(connector);
    }
    
    bool ensureLiveCacheReady(const std::string& instrument = "EUR_USD");
    
    std::vector<sep::connectors::MarketData> getLiveCachedData(const std::string& instrument = "EUR_USD");
    
    bool refreshLiveCache(const std::string& instrument = "EUR_USD");
    
    CacheInfo getLiveCacheInfo(const std::string& instrument = "EUR_USD");

    bool needsRefresh(const std::string& instrument = "EUR_USD") const;

    // ============ TRAINING DATA SYNC API (from RemoteDataManager) ============
    
    std::future<std::vector<TrainingData>> fetch_training_data(
        const std::string& pair, 
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    );
    
    std::future<bool> upload_training_batch(const std::vector<TrainingData>& batch);
    
    std::future<bool> upload_model(const ModelState& model);
    std::future<ModelState> download_latest_model(const std::string& pair);
    std::future<std::vector<ModelState>> list_available_models();
    
    void start_streaming(const std::vector<std::string>& pairs);
    void stop_streaming();
    bool register_data_callback(std::function<void(const TrainingData&)> callback);
    
    void clear_local_cache();
    size_t get_cache_size();
    bool is_cache_valid(const std::string& key);
    
    bool test_connection();
    nlohmann::json get_remote_status();

    // ============ UNIFIED API ============
    
    struct UnifiedCacheStatus {
        CacheInfo live_cache;
        bool remote_available;
        size_t training_data_count;
        size_t model_count;
    };
    
    UnifiedCacheStatus getUnifiedStatus();
    
    void cleanupCache();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace sep::trading
