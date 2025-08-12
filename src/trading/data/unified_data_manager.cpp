#include "unified_data_manager.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <thread>
#include <condition_variable>

#include "common/sep_precompiled.h"
#include "../_sep/testbed/placeholder_detection.h"
#include "common/financial_data_types.h"

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
    
    if (needsRefresh(instrument)) {
        return refreshLiveCache(instrument);
    }
    
    auto cache_info = getLiveCacheInfo(instrument);
    if (!cache_info.valid || cache_info.data_points < EXPECTED_48H_POINTS * 0.95) {
        return refreshLiveCache(instrument);
    }
    
    return true;
}

std::vector<sep::connectors::MarketData> UnifiedDataManager::getLiveCachedData(const std::string& instrument) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<sep::connectors::MarketData> data;
    if (!loadCacheFile(getLiveCacheFile(instrument), data)) {
        return {};
    }
    
    return data;
}

bool UnifiedDataManager::refreshLiveCache(const std::string& instrument) {
    if (!oanda_connector_) {
        return false;
    }
    
    std::vector<sep::connectors::OandaCandle> candles;
    std::mutex data_mutex;
    std::condition_variable data_ready;
    bool data_received = false;

    try {
        candles = oanda_connector_->getHistoricalData(instrument, "M1", "", "");
        data_received = true;
        data_ready.notify_one();

        std::unique_lock<std::mutex> lock(data_mutex);
        if (!data_ready.wait_for(lock, std::chrono::seconds(60), [&]{ return data_received; })) {
            return false;
        }
        
    } catch (const std::exception& e) {
        return false;
    }
    
    if (candles.empty()) {
        return false;
    }
    
    auto market_data = convertOandaCandlesToMarketData(candles, instrument);
    
    if (!saveCacheFile(getLiveCacheFile(instrument), market_data)) {
        return false;
    }
    
    return true;
}

CacheInfo UnifiedDataManager::getLiveCacheInfo(const std::string& instrument) {
    CacheInfo info{};
    info.cache_file = getLiveCacheFile(instrument);
    info.valid = false;
    info.data_points = 0;
    
    try {
        if (!std::filesystem::exists(info.cache_file)) {
            return info;
        }
        
        auto file_time = std::filesystem::last_write_time(info.cache_file);
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            file_time - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
        info.last_update = sctp;
        
        std::ifstream file(info.cache_file);
        if (file.is_open()) {
            nlohmann::json cache_json;
            file >> cache_json;
            
            if (cache_json.contains("data") && cache_json["data"].is_array()) {
                info.data_points = cache_json["data"].size();
                info.valid = true;
            }
        }
        
    } catch (const std::exception& e) {
    }
    
    return info;
}

bool UnifiedDataManager::needsRefresh(const std::string& instrument) const {
    auto cache_info = getLiveCacheInfo(instrument);
    
    if (!cache_info.valid) {
        return true;
    }
    
    auto age = std::chrono::system_clock::now() - cache_info.last_update;
    if (age > CACHE_REFRESH_INTERVAL) {
        return true;
    }
    
    if (cache_info.data_points < EXPECTED_48H_POINTS * 0.95) {
        return true;
    }
    
    return false;
}

// ============ TRAINING DATA SYNC API ============

std::future<std::vector<TrainingData>> UnifiedDataManager::fetchTrainingDataAsync(
    const std::string& pair, 
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end) {
    
    return std::async(std::launch::async, [this, pair, start, end]() {
        std::vector<TrainingData> result;
        
        // TODO: Implement training data fetching from remote
        
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
    
    status.live_cache = getLiveCacheInfo("EUR_USD");
    
    status.remote_available = isRemoteAvailable();
    
    // TODO: Count training data and models
    status.training_data_count = 0;
    status.model_count = 0;
    
    return status;
}

void UnifiedDataManager::cleanupCache() {
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
    }
}

// ============ PRIVATE HELPERS ============

std::string UnifiedDataManager::getLiveCacheFile(const std::string& instrument) const {
    return config_.live_cache_path + "/" + instrument + "_48h_cache.json";
}

std::string UnifiedDataManager::getTrainingCacheFile(const std::string& pair) const {
    return config_.local_cache_path + "/" + pair + ".cache";
}

bool UnifiedDataManager::validateCacheFile(const std::string& filepath, int ttl_hours) const {
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
        nlohmann::json cache_json;
        cache_json["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        cache_json["data_points"] = data.size();
        
        nlohmann::json data_array = nlohmann::json::array();
        for (const auto& md : data) {
            nlohmann::json item;
            item["instrument"] = md.instrument;
            item["mid"] = md.mid;
            item["bid"] = md.bid;
            item["ask"] = md.ask;
            item["volume"] = md.volume;
            item["atr"] = md.atr;
            item["timestamp"] = md.timestamp;
            
            data_array.push_back(item);
        }
        cache_json["data"] = data_array;
        
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        file << cache_json.dump(2);
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<sep::connectors::MarketData> UnifiedDataManager::loadCacheFile(const std::string& filepath) {
    std::vector<sep::connectors::MarketData> result;
    
    try {
        if (!std::filesystem::exists(filepath)) {
            return result;
        }
        
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return result;
        }
        
        nlohmann::json cache_json;
        file >> cache_json;
        
        if (!cache_json.contains("data") || !cache_json["data"].is_array()) {
            return result;
        }
        
        for (const auto& item : cache_json["data"]) {
            sep::connectors::MarketData md;
            md.instrument = item.value("instrument", "");
            md.mid = item.value("mid", 0.0);
            md.bid = item.value("bid", 0.0);
            md.ask = item.value("ask", 0.0);
            md.volume = item.value("volume", 0);
            md.atr = item.value("atr", 0.0001);
            md.timestamp = item.value("timestamp", 0ULL);
            
            result.push_back(md);
        }
        
    } catch (const std::exception& e) {
    }
    
    return result;
}

std::vector<sep::connectors::MarketData> UnifiedDataManager::convertOandaCandlesToMarketData(
    const std::vector<sep::connectors::OandaCandle>& candles, 
    const std::string& instrument) {
    
    std::vector<sep::connectors::MarketData> market_data;
    market_data.reserve(candles.size());
    
    for (const auto& candle : candles) {
        sep::connectors::MarketData md;
        md.instrument = instrument;
        md.mid = candle.close;
        md.bid = candle.close - 0.00001;
        md.ask = candle.close + 0.00001;
        md.volume = candle.volume;
        md.atr = 0.0001;
        
        try {
            auto time_point = sep::common::parseTimestamp(candle.time);
            md.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                time_point.time_since_epoch()).count();
        } catch (const std::exception& e) {
            md.timestamp = 0;
        }
        
        market_data.push_back(md);
    }
    
    return market_data;
}

} // namespace sep::trading