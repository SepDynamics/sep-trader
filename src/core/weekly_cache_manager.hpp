#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sep::cache {

enum class WeeklyCacheStatus {
    CURRENT,        // Cache covers current week adequately
    STALE,          // Cache is older than one week
    PARTIAL,        // Cache has some current week data but incomplete
    MISSING,        // No cache data for current week
    BUILDING,       // Currently building/updating cache
    ERROR           // Error accessing or building cache
};

enum class UpdatePriority {
    LOW,
    NORMAL,
    HIGH,
    CRITICAL
};

struct WeeklyCacheRequirement {
    std::chrono::hours min_coverage{168}; // 7 days * 24 hours
    double min_data_quality{0.85}; // Minimum quality score
    bool require_current_week{true}; // Must include current week data
    bool require_complete_days{true}; // Must have complete trading days
    std::chrono::hours update_frequency{24}; // Check every 24 hours
    std::chrono::hours rebuild_threshold{48}; // Rebuild if data older than 48h
    
    WeeklyCacheRequirement() = default;
};

struct CacheOperationResult {
    bool success;
    WeeklyCacheStatus status;
    std::string message;
    std::chrono::duration<double> operation_time;
    size_t records_processed;
    std::vector<std::string> warnings;
    
    CacheOperationResult() : success(false), status(WeeklyCacheStatus::ERROR), 
                           operation_time(std::chrono::duration<double>::zero()), 
                           records_processed(0) {}
};

class WeeklyCacheManager {
public:
    WeeklyCacheManager();
    ~WeeklyCacheManager();
    
    // Core functionality
    WeeklyCacheStatus checkWeeklyCacheStatus(const std::string& pair_symbol) const;
    CacheOperationResult ensureWeeklyCache(const std::string& pair_symbol, UpdatePriority priority = UpdatePriority::NORMAL);
    CacheOperationResult forceRebuildCache(const std::string& pair_symbol);
    std::unordered_map<std::string, WeeklyCacheStatus> checkAllWeeklyCaches() const;
    std::vector<CacheOperationResult> ensureAllWeeklyCaches(UpdatePriority priority = UpdatePriority::NORMAL);
    
    // Configuration
    void setWeeklyCacheRequirement(const WeeklyCacheRequirement& requirement);
    WeeklyCacheRequirement getWeeklyCacheRequirement() const;
    void setCustomRequirementForPair(const std::string& pair, const WeeklyCacheRequirement& requirement);
    
    // Data source provider
    using DataSourceProvider = std::function<std::vector<std::string>(
        const std::string& pair_symbol,
        std::chrono::system_clock::time_point from,
        std::chrono::system_clock::time_point to)>;
    void setDataSourceProvider(const DataSourceProvider& provider);
    
    // Implementation
    CacheOperationResult buildWeeklyCache(const std::string& pair_symbol, bool force_rebuild = false);
    CacheOperationResult fetchAndCacheWeeklyData(const std::string& pair_symbol);
    CacheOperationResult updateIncrementalCache(const std::string& pair_symbol);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Helper functions
std::string weeklyCacheStatusToString(WeeklyCacheStatus status);
WeeklyCacheStatus stringToWeeklyCacheStatus(const std::string& status_str);
bool isWeeklyCacheReady(WeeklyCacheStatus status);
UpdatePriority calculateUpdatePriority(const std::string& pair_symbol, WeeklyCacheStatus status);

} // namespace sep::cache