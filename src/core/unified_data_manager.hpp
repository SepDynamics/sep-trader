#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

// Forward declaration instead of direct include to avoid path issues
namespace sep::connectors {
    class OandaConnector;
}

namespace sep::trading {

struct UnifiedDataConfig {
    std::string cache_dir = "cache/";
    std::string remote_endpoint = "";
    int connection_timeout = 30;
    bool enable_remote_sync = false;
    int sync_interval = 3600;
    bool enable_local_cache = true;
    size_t max_cache_size = 1024 * 1024 * 50; // 50 MB
};

struct CacheInfo {
    std::chrono::system_clock::time_point last_update;
    size_t data_points;
    std::string cache_file;
    bool valid;
};

class UnifiedDataManager {
public:
    struct UnifiedCacheStatus {
        CacheInfo live_cache;
        bool remote_available;
        size_t training_data_count;
        size_t model_count;
    };

    explicit UnifiedDataManager(const UnifiedDataConfig& config);
    
    // Explicitly declare destructor to ensure proper destruction of forward-declared Impl
    ~UnifiedDataManager();

    // Prevent copying and moving
    UnifiedDataManager(const UnifiedDataManager&) = delete;
    UnifiedDataManager& operator=(const UnifiedDataManager&) = delete;
    UnifiedDataManager(UnifiedDataManager&&) = delete;
    UnifiedDataManager& operator=(UnifiedDataManager&&) = delete;

    // Core functionality - connector required for proper initialization
    bool initialize(sep::connectors::OandaConnector* connector);
    
    // Test function to verify header changes are detected
    bool test_build_detection() { return true; }
    UnifiedCacheStatus getCacheStatus() const;
    bool saveCacheData(const std::string& symbol, const std::vector<double>& data);
    std::vector<double> loadCacheData(const std::string& symbol);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sep::trading