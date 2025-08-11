#include "unified_data_manager.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <thread>

#include "common/sep_precompiled.h"

namespace sep::trading {

UnifiedDataManager::UnifiedDataManager(const UnifiedDataConfig& config) 
    : config_(config) {
    
    // Create cache directories
    std::filesystem::create_directories(config_.local_cache_path);
    std::filesystem::create_directories(config_.live_cache_path);
}

UnifiedDataManager::~UnifiedDataManager() = default;

// ============ LIVE TRADING CACHE API ============

bool UnifiedDataManager::initializeLiveTrading(sep::connectors::OandaConnector* connector) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    oanda_connector_ = connector;
    return oanda_connector_ != nullptr;
}

bool UnifiedDataManager::ensureLiveCacheReady(const std::string& instrument) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    if (!oanda_connector_) {
        return false;
    }
    
    std::string cache_file = getLiveCacheFile(instrument);
    
    // Check if cache exists and is valid
    if (validateCacheFile(cache_file, config_.live_cache_ttl_hours)) {
        return true;
    }
    
    // Fetch fresh data from OANDA
    return refreshLiveCache(instrument);
}

std::vector<sep::connectors::MarketData> UnifiedDataManager::getLiveCachedData(const std::string& instrument) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string cache_file = getLiveCacheFile(instrument);
    if (!validateCacheFile(cache_file, config_.live_cache_ttl_hours)) {
        return {};
    }
    
    return loadCacheFile(cache_file);
}

bool UnifiedDataManager::refreshLiveCache(const std::string& instrument) {
    if (!oanda_connector_) {
        return false;
    }
    
    // Fetch 48H of data from OANDA
    auto end_time = std::chrono::system_clock::now();
    auto start_time = end_time - std::chrono::hours(48);
    
    // TODO: Implement OANDA data fetching
    // For now, return true to avoid compilation issues
    // std::vector<sep::connectors::MarketData> data = oanda_connector_->fetchHistoricalData(instrument, start_time, end_time);
    
    std::string cache_file = getLiveCacheFile(instrument);
    // return saveCacheFile(cache_file, data);
    
    return true; // Placeholder
}

CacheInfo UnifiedDataManager::getLiveCacheInfo(const std::string& instrument) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    CacheInfo info{};
    std::string cache_file = getLiveCacheFile(instrument);
    
    if (std::filesystem::exists(cache_file)) {
        auto file_time = std::filesystem::last_write_time(cache_file);
        auto system_time = std::chrono::file_clock::to_sys(file_time);
        info.last_update = system_time;
        info.cache_file = cache_file;
        info.valid = validateCacheFile(cache_file, config_.live_cache_ttl_hours);
        
        // Count data points if file is valid
        if (info.valid) {
            auto data = loadCacheFile(cache_file);
            info.data_points = data.size();
        }
    }
    
    return info;
}

// ============ TRAINING DATA SYNC API ============

std::future<std::vector<TrainingData>> UnifiedDataManager::fetchTrainingDataAsync(
    const std::string& pair, 
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end) {
    
    return std::async(std::launch::async, [this, pair, start, end]() {
        std::vector<TrainingData> result;
        
        // TODO: Implement training data fetching from remote
        // This would involve PostgreSQL/Redis connection
        
        return result;
    });
}

std::future<bool> UnifiedDataManager::pushModelAsync(const ModelState& model) {
    return std::async(std::launch::async, [this, model]() {
        // TODO: Implement model pushing to remote
        return true; // Placeholder
    });
}

std::future<std::optional<ModelState>> UnifiedDataManager::pullLatestModelAsync(const std::string& pair) {
    return std::async(std::launch::async, [this, pair]() {
        std::optional<ModelState> result;
        
        // TODO: Implement model pulling from remote
        
        return result;
    });
}

std::future<bool> UnifiedDataManager::syncToRemoteAsync() {
    return std::async(std::launch::async, [this]() {
        // TODO: Implement sync to remote
        return true; // Placeholder
    });
}

bool UnifiedDataManager::isRemoteAvailable() {
    std::lock_guard<std::mutex> lock(remote_mutex_);
    return remote_available_;
}

// ============ UNIFIED API ============

UnifiedDataManager::UnifiedCacheStatus UnifiedDataManager::getUnifiedStatus() {
    UnifiedCacheStatus status{};
    
    // Get live cache status
    status.live_cache = getLiveCacheInfo("EUR_USD"); // Default instrument
    
    // Get remote status
    status.remote_available = isRemoteAvailable();
    
    // TODO: Count training data and models
    status.training_data_count = 0;
    status.model_count = 0;
    
    return status;
}

void UnifiedDataManager::cleanupCache() {
    // Clean up old live cache files
    try {
        for (const auto& entry : std::filesystem::directory_iterator(config_.live_cache_path)) {
            if (entry.is_regular_file()) {
                auto file_time = std::filesystem::last_write_time(entry);
                auto system_time = std::chrono::file_clock::to_sys(file_time);
                auto age = std::chrono::system_clock::now() - system_time;
                
                if (age > std::chrono::hours(config_.live_cache_ttl_hours * 2)) {
                    std::filesystem::remove(entry);
                }
            }
        }
        
        // Clean up old training cache files
        for (const auto& entry : std::filesystem::directory_iterator(config_.local_cache_path)) {
            if (entry.is_regular_file()) {
                auto file_time = std::filesystem::last_write_time(entry);
                auto system_time = std::chrono::file_clock::to_sys(file_time);
                auto age = std::chrono::system_clock::now() - system_time;
                
                if (age > std::chrono::hours(config_.cache_ttl_hours * 2)) {
                    std::filesystem::remove(entry);
                }
            }
        }
    } catch (const std::exception& e) {
        // Log error but don't fail
    }
}

// ============ PRIVATE HELPERS ============

std::string UnifiedDataManager::getLiveCacheFile(const std::string& instrument) {
    return config_.live_cache_path + "/live_" + instrument + ".cache";
}

std::string UnifiedDataManager::getTrainingCacheFile(const std::string& pair) {
    return config_.local_cache_path + "/training_" + pair + ".cache";
}

bool UnifiedDataManager::validateCacheFile(const std::string& filepath, int ttl_hours) {
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    auto file_time = std::filesystem::last_write_time(filepath);
    auto system_time = std::chrono::file_clock::to_sys(file_time);
    auto age = std::chrono::system_clock::now() - system_time;
    
    return age < std::chrono::hours(ttl_hours);
}

bool UnifiedDataManager::saveCacheFile(const std::string& filepath, const std::vector<sep::connectors::MarketData>& data) {
    try {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            return false;
        }
        
        // Simple binary serialization
        size_t count = data.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (const auto& item : data) {
            // TODO: Implement proper MarketData serialization
            // For now, just write placeholder
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<sep::connectors::MarketData> UnifiedDataManager::loadCacheFile(const std::string& filepath) {
    std::vector<sep::connectors::MarketData> result;
    
    try {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            return result;
        }
        
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        // TODO: Implement proper MarketData deserialization
        // For now, return empty vector
        
    } catch (const std::exception& e) {
        // Return empty vector on error
    }
    
    return result;
}

} // namespace sep::trading
