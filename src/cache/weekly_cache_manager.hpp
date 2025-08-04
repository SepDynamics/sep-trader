#pragma once

#include "cache_validator.hpp"
#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>

namespace sep::cache {

// Weekly cache management status
enum class WeeklyCacheStatus {
    CURRENT,        // Cache covers current week adequately
    STALE,          // Cache is older than one week
    PARTIAL,        // Cache has some current week data but incomplete
    MISSING,        // No cache data for current week
    BUILDING,       // Currently building/updating cache
    ERROR          // Error accessing or building cache
};

// Weekly cache requirements
struct WeeklyCacheRequirement {
    std::chrono::hours min_coverage{168}; // 7 days * 24 hours
    double min_data_quality{0.85}; // Minimum quality score
    bool require_current_week{true}; // Must include current week data
    bool require_complete_days{true}; // Must have complete trading days
    std::chrono::hours update_frequency{24}; // Check every 24 hours
    std::chrono::hours rebuild_threshold{48}; // Rebuild if data older than 48h
    
    WeeklyCacheRequirement() = default;
};

// Cache update priority
enum class UpdatePriority {
    LOW,        // Background update when convenient
    NORMAL,     // Standard priority update
    HIGH,       // Important pair, update soon
    CRITICAL,   // Trading depends on this, update immediately
    EMERGENCY   // System failure, rebuild immediately
};

// Cache operation result
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

// Weekly cache update callback types
using CacheUpdateCallback = std::function<void(const std::string& pair, WeeklyCacheStatus old_status, WeeklyCacheStatus new_status)>;
using CacheBuildCallback = std::function<void(const std::string& pair, const CacheOperationResult& result)>;
using CacheErrorCallback = std::function<void(const std::string& pair, const std::string& error_message)>;

class WeeklyCacheManager {
public:
    WeeklyCacheManager();
    ~WeeklyCacheManager();

    // Primary cache management
    WeeklyCacheStatus checkWeeklyCacheStatus(const std::string& pair_symbol) const;
    CacheOperationResult ensureWeeklyCache(const std::string& pair_symbol, UpdatePriority priority = UpdatePriority::NORMAL);
    CacheOperationResult forceRebuildCache(const std::string& pair_symbol);
    
    // Batch operations
    std::unordered_map<std::string, WeeklyCacheStatus> checkAllWeeklyCaches() const;
    std::vector<CacheOperationResult> ensureAllWeeklyCaches(UpdatePriority priority = UpdatePriority::NORMAL);
    std::vector<std::string> getStaleCachePairs() const;
    std::vector<std::string> getMissingCachePairs() const;
    
    // Cache requirements management
    void setWeeklyCacheRequirement(const WeeklyCacheRequirement& requirement);
    WeeklyCacheRequirement getWeeklyCacheRequirement() const;
    void setCustomRequirementForPair(const std::string& pair, const WeeklyCacheRequirement& requirement);
    bool hasCustomRequirement(const std::string& pair) const;
    
    // Automatic management
    void enableAutomaticManagement(bool enable = true);
    bool isAutomaticManagementEnabled() const;
    void setCheckInterval(std::chrono::minutes interval);
    std::chrono::minutes getCheckInterval() const;
    
    // Trading integration
    bool canPairTrade(const std::string& pair_symbol) const;
    std::vector<std::string> getTradingReadyPairs() const;
    std::vector<std::string> getPairsRequiringUpdate() const;
    bool blockTradeIfCacheStale(const std::string& pair_symbol) const;
    
    // Cache building and updating
    CacheOperationResult buildWeeklyCache(const std::string& pair_symbol, bool force_rebuild = false);
    CacheOperationResult updateIncrementalCache(const std::string& pair_symbol);
    CacheOperationResult repairCache(const std::string& pair_symbol);
    
    // Data source integration
    void setDataSourceProvider(std::function<std::vector<std::string>(const std::string&, std::chrono::system_clock::time_point, std::chrono::system_clock::time_point)> provider);
    bool hasDataSourceProvider() const;
    CacheOperationResult fetchAndCacheWeeklyData(const std::string& pair_symbol);
    
    // Cache maintenance
    void performMaintenanceCleanup();
    void removeOldCacheFiles(std::chrono::hours max_age = std::chrono::hours(720)); // 30 days
    void compactCacheFiles();
    void validateAndRepairAllCaches();
    
    // Event system
    size_t addUpdateCallback(CacheUpdateCallback callback);
    void removeUpdateCallback(size_t callback_id);
    size_t addBuildCallback(CacheBuildCallback callback);
    void removeBuildCallback(size_t callback_id);
    size_t addErrorCallback(CacheErrorCallback callback);
    void removeErrorCallback(size_t callback_id);
    
    // Priority queue management
    void addToUpdateQueue(const std::string& pair_symbol, UpdatePriority priority);
    void removeFromUpdateQueue(const std::string& pair_symbol);
    std::vector<std::pair<std::string, UpdatePriority>> getUpdateQueue() const;
    bool isInUpdateQueue(const std::string& pair_symbol) const;
    
    // Statistics and monitoring
    size_t getTotalCachesManaged() const;
    size_t getCurrentWeeklyCaches() const;
    size_t getStaleCaches() const;
    double getAverageCacheQuality() const;
    std::chrono::system_clock::time_point getLastMaintenanceTime() const;
    std::chrono::duration<double> getAverageUpdateTime() const;
    
    // Configuration
    void setCacheDirectory(const std::string& directory);
    std::string getCacheDirectory() const;
    void setMaxConcurrentUpdates(size_t max_concurrent);
    size_t getMaxConcurrentUpdates() const;
    
    // Week calculation utilities
    std::chrono::system_clock::time_point getCurrentWeekStart() const;
    std::chrono::system_clock::time_point getCurrentWeekEnd() const;
    std::chrono::system_clock::time_point getLastWeekStart() const;
    std::chrono::system_clock::time_point getLastWeekEnd() const;
    bool isCurrentWeek(std::chrono::system_clock::time_point timestamp) const;
    bool isLastWeek(std::chrono::system_clock::time_point timestamp) const;

private:
    mutable std::mutex manager_mutex_;
    WeeklyCacheRequirement default_requirement_;
    std::unordered_map<std::string, WeeklyCacheRequirement> pair_requirements_;
    std::string cache_directory_;
    
    // Automatic management
    std::atomic<bool> automatic_management_enabled_{true};
    std::chrono::minutes check_interval_{15}; // Check every 15 minutes
    std::unique_ptr<std::thread> management_thread_;
    std::atomic<bool> stop_management_{false};
    std::condition_variable management_cv_;
    
    // Update queue and processing
    struct QueueEntry {
        std::string pair_symbol;
        UpdatePriority priority;
        std::chrono::system_clock::time_point queued_time;
        
        bool operator<(const QueueEntry& other) const {
            return priority < other.priority; // Higher priority values come first
        }
    };
    
    std::vector<QueueEntry> update_queue_;
    std::atomic<size_t> max_concurrent_updates_{3};
    std::atomic<size_t> active_updates_{0};
    std::unordered_map<std::string, std::unique_ptr<std::thread>> active_update_threads_;
    mutable std::mutex queue_mutex_;
    
    // Data source provider
    std::function<std::vector<std::string>(const std::string&, std::chrono::system_clock::time_point, std::chrono::system_clock::time_point)> data_source_provider_;
    mutable std::mutex data_source_mutex_;
    
    // Event callbacks
    std::vector<CacheUpdateCallback> update_callbacks_;
    std::vector<CacheBuildCallback> build_callbacks_;
    std::vector<CacheErrorCallback> error_callbacks_;
    mutable std::mutex callbacks_mutex_;
    
    // Statistics
    mutable std::atomic<size_t> total_updates_performed_{0};
    mutable std::atomic<size_t> successful_updates_{0};
    mutable std::atomic<size_t> failed_updates_{0};
    mutable std::atomic<std::chrono::duration<double>> total_update_time_{std::chrono::duration<double>::zero()};
    std::chrono::system_clock::time_point last_maintenance_time_;
    
    // Cache validator integration
    std::unique_ptr<CacheValidator> cache_validator_;
    
    // Internal management methods
    void managementLoop();
    void processUpdateQueue();
    void performPeriodicChecks();
    WeeklyCacheStatus assessCacheStatus(const std::string& pair_symbol) const;
    
    // Cache building implementation
    CacheOperationResult performCacheBuild(const std::string& pair_symbol, bool force_rebuild);
    CacheOperationResult performIncrementalUpdate(const std::string& pair_symbol);
    bool mergeCacheData(const std::string& existing_cache, const std::vector<std::string>& new_data);
    
    // Queue management
    void processQueueEntry(const QueueEntry& entry);
    void sortUpdateQueue();
    QueueEntry* findQueueEntry(const std::string& pair_symbol);
    
    // File operations
    std::string getWeeklyCachePath(const std::string& pair_symbol) const;
    std::string getBackupCachePath(const std::string& pair_symbol) const;
    bool createCacheDirectory() const;
    bool backupExistingCache(const std::string& pair_symbol) const;
    bool restoreFromBackup(const std::string& pair_symbol) const;
    
    // Data processing
    std::vector<std::string> fetchDataForWeek(const std::string& pair_symbol, 
                                            std::chrono::system_clock::time_point week_start) const;
    bool writeCacheFile(const std::string& cache_path, const std::vector<std::string>& data) const;
    std::vector<std::string> readCacheFile(const std::string& cache_path) const;
    
    // Event notification
    void notifyUpdateStatus(const std::string& pair, WeeklyCacheStatus old_status, WeeklyCacheStatus new_status);
    void notifyBuildResult(const std::string& pair, const CacheOperationResult& result);
    void notifyError(const std::string& pair, const std::string& error_message);
    
    // Utility methods
    bool isWeekendDay(std::chrono::system_clock::time_point timestamp) const;
    std::chrono::system_clock::time_point getWeekStart(std::chrono::system_clock::time_point timestamp) const;
    std::chrono::system_clock::time_point addDays(std::chrono::system_clock::time_point timestamp, int days) const;
    UpdatePriority stringToPriority(const std::string& priority_str) const;
    std::string priorityToString(UpdatePriority priority) const;
};

// Utility functions
std::string weeklyCacheStatusToString(WeeklyCacheStatus status);
WeeklyCacheStatus stringToWeeklyCacheStatus(const std::string& status_str);
bool isWeeklyCacheReady(WeeklyCacheStatus status);
UpdatePriority calculateUpdatePriority(const std::string& pair_symbol, WeeklyCacheStatus status);

// Global weekly cache manager instance
WeeklyCacheManager& getGlobalWeeklyCacheManager();

} // namespace sep::cache
