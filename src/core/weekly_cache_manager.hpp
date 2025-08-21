#pragma once

#include <string>
#include <functional>
#include <chrono>
#include <vector>
#include <unordered_map>

namespace sep::cache {

enum class WeeklyCacheStatus {
    CURRENT,        // Cache covers current week adequately
    STALE,          // Cache is older than one week
    PARTIAL,        // Cache has some current week data but incomplete
    MISSING,        // No cache data for current week
    BUILDING,       // Currently building/updating cache
    ERROR           // Error accessing or building cache
};

class WeeklyCacheManager {
public:
    WeeklyCacheManager() = default;
    ~WeeklyCacheManager() = default;
    
    // Data source provider setup
    void setDataSourceProvider(
        std::function<std::vector<std::string>(
            const std::string& pair_symbol,
            std::chrono::system_clock::time_point from,
            std::chrono::system_clock::time_point to
        )> provider) {
        // Store the provider function for data source integration
        data_source_provider_ = std::move(provider);
    }
    
    // Check cache status for specific pair symbol
    WeeklyCacheStatus getCacheStatus(const std::string& pair_symbol) {
        // Check if pair symbol is in our cache map
        auto it = cache_status_map_.find(pair_symbol);
        if (it == cache_status_map_.end()) {
            // Check if we have valid data source
            if (data_source_provider_) {
                // Try to get data to determine status using simple time points
                try {
                    auto now = std::chrono::system_clock::now();
                    // Use time_t for simpler arithmetic - 7 days in seconds
                    auto now_time_t = std::chrono::system_clock::to_time_t(now);
                    auto week_ago_time_t = now_time_t - (7 * 24 * 3600); // 7 days in seconds
                    auto week_ago = std::chrono::system_clock::from_time_t(week_ago_time_t);
                    
                    auto data = data_source_provider_(pair_symbol, week_ago, now);
                    
                    if (data.empty()) {
                        cache_status_map_[pair_symbol] = WeeklyCacheStatus::MISSING;
                        return WeeklyCacheStatus::MISSING;
                    } else if (data.size() < 100) { // Heuristic for partial data
                        cache_status_map_[pair_symbol] = WeeklyCacheStatus::PARTIAL;
                        return WeeklyCacheStatus::PARTIAL;
                    } else {
                        cache_status_map_[pair_symbol] = WeeklyCacheStatus::CURRENT;
                        return WeeklyCacheStatus::CURRENT;
                    }
                } catch (const std::exception&) {
                    cache_status_map_[pair_symbol] = WeeklyCacheStatus::ERROR;
                    return WeeklyCacheStatus::ERROR;
                }
            }
            // Default to missing if no data source
            return WeeklyCacheStatus::MISSING;
        }
        
        return it->second;
    }
    
    bool isValidCache(const std::string& pair_symbol) {
        WeeklyCacheStatus status = getCacheStatus(pair_symbol);
        return status == WeeklyCacheStatus::CURRENT || status == WeeklyCacheStatus::PARTIAL;
    }
    
    int getManagedPairCount() {
        return static_cast<int>(cache_status_map_.size());
    }

private:
    std::function<std::vector<std::string>(
        const std::string&,
        std::chrono::system_clock::time_point,
        std::chrono::system_clock::time_point
    )> data_source_provider_;
    
    // In-memory cache status tracking
    std::unordered_map<std::string, WeeklyCacheStatus> cache_status_map_;
};

} // namespace sep::cache