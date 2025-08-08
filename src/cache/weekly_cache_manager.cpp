#include "weekly_cache_manager.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <queue>
#include <sstream>

#include "../nlohmann_json_protected.h"

namespace sep::cache {

namespace fs = std::filesystem;

WeeklyCacheManager::WeeklyCacheManager() 
    : cache_directory_("cache/weekly/"),
      cache_validator_(std::make_unique<CacheValidator>()) {
    
    // Ensure cache directory exists
    createCacheDirectory();
    
    // Configure cache validator for weekly cache requirements
    cache_validator_->setCacheBasePath(cache_directory_);
    
    spdlog::info("WeeklyCacheManager initialized with directory: {}", cache_directory_);
}

WeeklyCacheManager::~WeeklyCacheManager() {
    if (management_thread_ && management_thread_->joinable()) {
        stop_management_ = true;
        management_cv_.notify_all();
        management_thread_->join();
    }
}

WeeklyCacheStatus WeeklyCacheManager::checkWeeklyCacheStatus(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return assessCacheStatus(pair_symbol);
}

CacheOperationResult WeeklyCacheManager::ensureWeeklyCache(const std::string& pair_symbol, UpdatePriority priority) {
    auto status = checkWeeklyCacheStatus(pair_symbol);
    
    if (status == WeeklyCacheStatus::CURRENT) {
        CacheOperationResult result;
        result.success = true;
        result.status = status;
        result.message = "Cache is already current";
        return result;
    }
    
    // Add to update queue if not current
    addToUpdateQueue(pair_symbol, priority);
    
    // If high priority, process immediately
    if (priority >= UpdatePriority::HIGH) {
        return buildWeeklyCache(pair_symbol, false);
    }
    
    CacheOperationResult result;
    result.success = true;
    result.status = WeeklyCacheStatus::BUILDING;
    result.message = "Added to update queue";
    return result;
}

CacheOperationResult WeeklyCacheManager::forceRebuildCache(const std::string& pair_symbol) {
    spdlog::info("Force rebuilding weekly cache for pair: {}", pair_symbol);
    return buildWeeklyCache(pair_symbol, true);
}

std::unordered_map<std::string, WeeklyCacheStatus> WeeklyCacheManager::checkAllWeeklyCaches() const {
    std::unordered_map<std::string, WeeklyCacheStatus> results;
    
    if (!fs::exists(cache_directory_)) {
        return results;
    }
    
    for (const auto& entry : fs::directory_iterator(cache_directory_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            std::string filename = entry.path().stem().string();
            // Extract pair symbol from filename (assuming format: "PAIR_weekly.json")
            if (filename.length() >= 7 && filename.substr(filename.length() - 7) == "_weekly") {
                std::string pair = filename.substr(0, filename.length() - 7);
                results[pair] = checkWeeklyCacheStatus(pair);
            }
        }
    }
    
    return results;
}

std::vector<CacheOperationResult> WeeklyCacheManager::ensureAllWeeklyCaches(UpdatePriority priority) {
    std::vector<CacheOperationResult> results;
    auto status_map = checkAllWeeklyCaches();
    
    for (const auto& [pair, status] : status_map) {
        if (status != WeeklyCacheStatus::CURRENT) {
            results.push_back(ensureWeeklyCache(pair, priority));
        }
    }
    
    return results;
}

std::vector<std::string> WeeklyCacheManager::getStaleCachePairs() const {
    std::vector<std::string> stale_pairs;
    auto status_map = checkAllWeeklyCaches();
    
    for (const auto& [pair, status] : status_map) {
        if (status == WeeklyCacheStatus::STALE) {
            stale_pairs.push_back(pair);
        }
    }
    
    return stale_pairs;
}

std::vector<std::string> WeeklyCacheManager::getMissingCachePairs() const {
    std::vector<std::string> missing_pairs;
    auto status_map = checkAllWeeklyCaches();
    
    for (const auto& [pair, status] : status_map) {
        if (status == WeeklyCacheStatus::MISSING) {
            missing_pairs.push_back(pair);
        }
    }
    
    return missing_pairs;
}

void WeeklyCacheManager::setWeeklyCacheRequirement(const WeeklyCacheRequirement& requirement) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    default_requirement_ = requirement;
    spdlog::info("Updated default weekly cache requirement");
}

WeeklyCacheRequirement WeeklyCacheManager::getWeeklyCacheRequirement() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return default_requirement_;
}

void WeeklyCacheManager::setCustomRequirementForPair(const std::string& pair, const WeeklyCacheRequirement& requirement) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    pair_requirements_[pair] = requirement;
    spdlog::info("Set custom weekly cache requirement for pair: {}", pair);
}

bool WeeklyCacheManager::hasCustomRequirement(const std::string& pair) const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return pair_requirements_.find(pair) != pair_requirements_.end();
}

void WeeklyCacheManager::enableAutomaticManagement(bool enable) {
    automatic_management_enabled_ = enable;
    
    if (enable && !management_thread_) {
        stop_management_ = false;
        management_thread_ = std::make_unique<std::thread>(&WeeklyCacheManager::managementLoop, this);
        spdlog::info("Automatic weekly cache management enabled");
    } else if (!enable && management_thread_) {
        stop_management_ = true;
        management_cv_.notify_all();
        if (management_thread_->joinable()) {
            management_thread_->join();
        }
        management_thread_.reset();
        spdlog::info("Automatic weekly cache management disabled");
    }
}

bool WeeklyCacheManager::isAutomaticManagementEnabled() const {
    return automatic_management_enabled_;
}

void WeeklyCacheManager::setCheckInterval(std::chrono::minutes interval) {
    check_interval_ = interval;
    management_cv_.notify_all(); // Wake up management thread with new interval
    spdlog::info("Weekly cache check interval set to {} minutes", interval.count());
}

std::chrono::minutes WeeklyCacheManager::getCheckInterval() const {
    return check_interval_;
}

bool WeeklyCacheManager::canPairTrade(const std::string& pair_symbol) const {
    auto status = checkWeeklyCacheStatus(pair_symbol);
    return status == WeeklyCacheStatus::CURRENT;
}

std::vector<std::string> WeeklyCacheManager::getTradingReadyPairs() const {
    std::vector<std::string> ready_pairs;
    auto status_map = checkAllWeeklyCaches();
    
    for (const auto& [pair, status] : status_map) {
        if (status == WeeklyCacheStatus::CURRENT) {
            ready_pairs.push_back(pair);
        }
    }
    
    return ready_pairs;
}

std::vector<std::string> WeeklyCacheManager::getPairsRequiringUpdate() const {
    std::vector<std::string> update_pairs;
    auto status_map = checkAllWeeklyCaches();
    
    for (const auto& [pair, status] : status_map) {
        if (status != WeeklyCacheStatus::CURRENT && status != WeeklyCacheStatus::BUILDING) {
            update_pairs.push_back(pair);
        }
    }
    
    return update_pairs;
}

bool WeeklyCacheManager::blockTradeIfCacheStale(const std::string& pair_symbol) const {
    return !canPairTrade(pair_symbol);
}

CacheOperationResult WeeklyCacheManager::buildWeeklyCache(const std::string& pair_symbol, bool force_rebuild) {
    auto start_time = std::chrono::steady_clock::now();
    
    CacheOperationResult result;
    result.status = WeeklyCacheStatus::BUILDING;
    
    try {
        spdlog::info("Building weekly cache for pair: {} (force={})", pair_symbol, force_rebuild);
        
        std::string cache_path = getWeeklyCachePath(pair_symbol);
        
        // Backup existing cache if it exists and we're not forcing rebuild
        if (!force_rebuild && fs::exists(cache_path)) {
            backupExistingCache(pair_symbol);
        }
        
        // Fetch weekly data
        auto week_start = getCurrentWeekStart();
        std::vector<std::string> weekly_data = fetchDataForWeek(pair_symbol, week_start);
        
        if (weekly_data.empty()) {
            result.success = false;
            result.status = WeeklyCacheStatus::ERROR;
            result.message = "No data available for current week";
            result.warnings.push_back("Failed to fetch weekly data");
            return result;
        }
        
        // Write cache file
        if (!writeCacheFile(cache_path, weekly_data)) {
            result.success = false;
            result.status = WeeklyCacheStatus::ERROR;
            result.message = "Failed to write cache file";
            return result;
        }
        
        result.success = true;
        result.status = WeeklyCacheStatus::CURRENT;
        result.message = "Weekly cache built successfully";
        result.records_processed = weekly_data.size();
        
        auto end_time = std::chrono::steady_clock::now();
        result.operation_time = std::chrono::duration<double>(end_time - start_time);
        
        // Update statistics
        total_updates_performed_++;
        successful_updates_++;
        total_update_time_.store(total_update_time_.load() + result.operation_time);
        
        // Notify callbacks
        notifyBuildResult(pair_symbol, result);
        
        spdlog::info("Weekly cache built for {} in {:.3f}ms ({} records)", 
                     pair_symbol, result.operation_time.count() * 1000, result.records_processed);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.status = WeeklyCacheStatus::ERROR;
        result.message = std::string("Exception during cache build: ") + e.what();
        
        failed_updates_++;
        notifyError(pair_symbol, result.message);
        
        spdlog::error("Failed to build weekly cache for {}: {}", pair_symbol, e.what());
    }
    
    return result;
}

CacheOperationResult WeeklyCacheManager::updateIncrementalCache(const std::string& pair_symbol) {
    spdlog::info("Performing incremental cache update for pair: {}", pair_symbol);
    return performIncrementalUpdate(pair_symbol);
}

CacheOperationResult WeeklyCacheManager::repairCache(const std::string& pair_symbol) {
    spdlog::info("Attempting to repair cache for pair: {}", pair_symbol);
    
    // Try to restore from backup first
    if (restoreFromBackup(pair_symbol)) {
        CacheOperationResult result;
        result.success = true;
        result.status = WeeklyCacheStatus::CURRENT;
        result.message = "Cache restored from backup";
        return result;
    }
    
    // If backup restore fails, rebuild from source
    return buildWeeklyCache(pair_symbol, true);
}

void WeeklyCacheManager::setDataSourceProvider(std::function<std::vector<std::string>(const std::string&, std::chrono::system_clock::time_point, std::chrono::system_clock::time_point)> provider) {
    std::lock_guard<std::mutex> lock(data_source_mutex_);
    data_source_provider_ = std::move(provider);
    spdlog::info("Data source provider configured");
}

bool WeeklyCacheManager::hasDataSourceProvider() const {
    std::lock_guard<std::mutex> lock(data_source_mutex_);
    return static_cast<bool>(data_source_provider_);
}

CacheOperationResult WeeklyCacheManager::fetchAndCacheWeeklyData(const std::string& pair_symbol) {
    if (!hasDataSourceProvider()) {
        CacheOperationResult result;
        result.success = false;
        result.status = WeeklyCacheStatus::ERROR;
        result.message = "No data source provider configured";
        return result;
    }
    
    return buildWeeklyCache(pair_symbol, false);
}

void WeeklyCacheManager::performMaintenanceCleanup() {
    spdlog::info("Performing weekly cache maintenance cleanup");
    
    removeOldCacheFiles();
    compactCacheFiles();
    validateAndRepairAllCaches();
    
    last_maintenance_time_ = std::chrono::system_clock::now();
    spdlog::info("Weekly cache maintenance completed");
}

void WeeklyCacheManager::removeOldCacheFiles(std::chrono::hours max_age) {
    if (!fs::exists(cache_directory_)) return;
    
    auto cutoff_time = std::chrono::system_clock::now() - max_age;
    size_t removed_count = 0;
    
    for (const auto& entry : fs::directory_iterator(cache_directory_)) {
        if (entry.is_regular_file()) {
            auto file_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                entry.last_write_time() - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
            
            if (file_time < cutoff_time) {
                try {
                    fs::remove(entry.path());
                    removed_count++;
                    spdlog::debug("Removed old cache file: {}", entry.path().string());
                } catch (const std::exception& e) {
                    spdlog::warn("Failed to remove old cache file {}: {}", entry.path().string(), e.what());
                }
            }
        }
    }
    
    spdlog::info("Removed {} old cache files older than {} hours", removed_count, max_age.count());
}

void WeeklyCacheManager::compactCacheFiles() {
    // Basic implementation - in a real system this might involve 
    // compressing files or optimizing their structure
    spdlog::debug("Cache file compaction completed (placeholder implementation)");
}

void WeeklyCacheManager::validateAndRepairAllCaches() {
    auto status_map = checkAllWeeklyCaches();
    size_t repaired_count = 0;
    
    for (const auto& [pair, status] : status_map) {
        if (status == WeeklyCacheStatus::ERROR) {
            auto repair_result = repairCache(pair);
            if (repair_result.success) {
                repaired_count++;
            }
        }
    }
    
    spdlog::info("Validated and repaired {} caches", repaired_count);
}

size_t WeeklyCacheManager::addUpdateCallback(CacheUpdateCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    update_callbacks_.push_back(std::move(callback));
    return update_callbacks_.size() - 1;
}

void WeeklyCacheManager::removeUpdateCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < update_callbacks_.size()) {
        update_callbacks_.erase(update_callbacks_.begin() + callback_id);
    }
}

size_t WeeklyCacheManager::addBuildCallback(CacheBuildCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    build_callbacks_.push_back(std::move(callback));
    return build_callbacks_.size() - 1;
}

void WeeklyCacheManager::removeBuildCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < build_callbacks_.size()) {
        build_callbacks_.erase(build_callbacks_.begin() + callback_id);
    }
}

size_t WeeklyCacheManager::addErrorCallback(CacheErrorCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    error_callbacks_.push_back(std::move(callback));
    return error_callbacks_.size() - 1;
}

void WeeklyCacheManager::removeErrorCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < error_callbacks_.size()) {
        error_callbacks_.erase(error_callbacks_.begin() + callback_id);
    }
}

void WeeklyCacheManager::addToUpdateQueue(const std::string& pair_symbol, UpdatePriority priority) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Remove existing entry if present
    auto it = std::find_if(update_queue_.begin(), update_queue_.end(),
                          [&pair_symbol](const QueueEntry& entry) {
                              return entry.pair_symbol == pair_symbol;
                          });
    
    if (it != update_queue_.end()) {
        // Update priority if higher
        if (priority > it->priority) {
            it->priority = priority;
            it->queued_time = std::chrono::system_clock::now();
        }
    } else {
        // Add new entry
        QueueEntry entry;
        entry.pair_symbol = pair_symbol;
        entry.priority = priority;
        entry.queued_time = std::chrono::system_clock::now();
        update_queue_.push_back(entry);
    }
    
    sortUpdateQueue();
    spdlog::debug("Added {} to update queue with priority {}", pair_symbol, static_cast<int>(priority));
}

void WeeklyCacheManager::removeFromUpdateQueue(const std::string& pair_symbol) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    auto it = std::find_if(update_queue_.begin(), update_queue_.end(),
                          [&pair_symbol](const QueueEntry& entry) {
                              return entry.pair_symbol == pair_symbol;
                          });
    
    if (it != update_queue_.end()) {
        update_queue_.erase(it);
        spdlog::debug("Removed {} from update queue", pair_symbol);
    }
}

std::vector<std::pair<std::string, UpdatePriority>> WeeklyCacheManager::getUpdateQueue() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    std::vector<std::pair<std::string, UpdatePriority>> queue_items;
    for (const auto& entry : update_queue_) {
        queue_items.emplace_back(entry.pair_symbol, entry.priority);
    }
    
    return queue_items;
}

bool WeeklyCacheManager::isInUpdateQueue(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    return std::any_of(update_queue_.begin(), update_queue_.end(),
                      [&pair_symbol](const QueueEntry& entry) {
                          return entry.pair_symbol == pair_symbol;
                      });
}

size_t WeeklyCacheManager::getTotalCachesManaged() const {
    return checkAllWeeklyCaches().size();
}

size_t WeeklyCacheManager::getCurrentWeeklyCaches() const {
    auto status_map = checkAllWeeklyCaches();
    return std::count_if(status_map.begin(), status_map.end(),
                        [](const auto& pair) {
                            return pair.second == WeeklyCacheStatus::CURRENT;
                        });
}

size_t WeeklyCacheManager::getStaleCaches() const {
    auto status_map = checkAllWeeklyCaches();
    return std::count_if(status_map.begin(), status_map.end(),
                        [](const auto& pair) {
                            return pair.second == WeeklyCacheStatus::STALE;
                        });
}

double WeeklyCacheManager::getAverageCacheQuality() const {
    auto status_map = checkAllWeeklyCaches();
    if (status_map.empty()) return 0.0;
    
    size_t good_caches = std::count_if(status_map.begin(), status_map.end(),
                                      [](const auto& pair) {
                                          return pair.second == WeeklyCacheStatus::CURRENT;
                                      });
    
    return static_cast<double>(good_caches) / status_map.size();
}

std::chrono::system_clock::time_point WeeklyCacheManager::getLastMaintenanceTime() const {
    return last_maintenance_time_;
}

std::chrono::duration<double> WeeklyCacheManager::getAverageUpdateTime() const {
    size_t total = total_updates_performed_;
    if (total == 0) return std::chrono::duration<double>::zero();
    return total_update_time_.load() / total;
}

void WeeklyCacheManager::setCacheDirectory(const std::string& directory) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    cache_directory_ = directory;
    if (!cache_directory_.empty() && cache_directory_.back() != '/') {
        cache_directory_ += '/';
    }
    createCacheDirectory();
    cache_validator_->setCacheBasePath(cache_directory_);
    spdlog::info("Weekly cache directory set to: {}", cache_directory_);
}

std::string WeeklyCacheManager::getCacheDirectory() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return cache_directory_;
}

void WeeklyCacheManager::setMaxConcurrentUpdates(size_t max_concurrent) {
    max_concurrent_updates_ = max_concurrent;
    spdlog::info("Max concurrent cache updates set to: {}", max_concurrent);
}

size_t WeeklyCacheManager::getMaxConcurrentUpdates() const {
    return max_concurrent_updates_;
}

std::chrono::system_clock::time_point WeeklyCacheManager::getCurrentWeekStart() const {
    auto now = std::chrono::system_clock::now();
    return getWeekStart(now);
}

std::chrono::system_clock::time_point WeeklyCacheManager::getCurrentWeekEnd() const {
    auto week_start = getCurrentWeekStart();
    return addDays(week_start, 7);
}

std::chrono::system_clock::time_point WeeklyCacheManager::getLastWeekStart() const {
    auto current_week_start = getCurrentWeekStart();
    return addDays(current_week_start, -7);
}

std::chrono::system_clock::time_point WeeklyCacheManager::getLastWeekEnd() const {
    auto last_week_start = getLastWeekStart();
    return addDays(last_week_start, 7);
}

bool WeeklyCacheManager::isCurrentWeek(std::chrono::system_clock::time_point timestamp) const {
    auto week_start = getCurrentWeekStart();
    auto week_end = getCurrentWeekEnd();
    return timestamp >= week_start && timestamp < week_end;
}

bool WeeklyCacheManager::isLastWeek(std::chrono::system_clock::time_point timestamp) const {
    auto week_start = getLastWeekStart();
    auto week_end = getLastWeekEnd();
    return timestamp >= week_start && timestamp < week_end;
}

// Private method implementations

void WeeklyCacheManager::managementLoop() {
    std::unique_lock<std::mutex> lock(manager_mutex_);
    
    while (!stop_management_) {
        performPeriodicChecks();
        processUpdateQueue();
        
        // Wait for the check interval or until notified to stop
        management_cv_.wait_for(lock, check_interval_, [this] { return stop_management_.load(); });
    }
}

void WeeklyCacheManager::processUpdateQueue() {
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    
    // Process queue entries while respecting concurrency limits
    while (!update_queue_.empty() && active_updates_ < max_concurrent_updates_) {
        auto entry = update_queue_.front();
        update_queue_.erase(update_queue_.begin());
        
        // Launch async update
        active_updates_++;
        active_update_threads_[entry.pair_symbol] = std::make_unique<std::thread>(
            &WeeklyCacheManager::processQueueEntry, this, entry);
    }
    
    // Clean up completed threads
    for (auto it = active_update_threads_.begin(); it != active_update_threads_.end();) {
        if (it->second->joinable()) {
            it->second->join();
            it = active_update_threads_.erase(it);
            active_updates_--;
        } else {
            ++it;
        }
    }
}

void WeeklyCacheManager::performPeriodicChecks() {
    auto pairs_needing_update = getPairsRequiringUpdate();
    
    for (const auto& pair : pairs_needing_update) {
        auto priority = calculateUpdatePriority(pair, checkWeeklyCacheStatus(pair));
        addToUpdateQueue(pair, priority);
    }
}

WeeklyCacheStatus WeeklyCacheManager::assessCacheStatus(const std::string& pair_symbol) const {
    std::string cache_path = getWeeklyCachePath(pair_symbol);
    
    // Check if cache file exists
    if (!fs::exists(cache_path)) {
        return WeeklyCacheStatus::MISSING;
    }
    
    // Validate cache using cache validator
    auto validation_result = cache_validator_->validateCache(cache_path);
    
    if (validation_result == ValidationResult::CORRUPTED) {
        return WeeklyCacheStatus::ERROR;
    }
    
    if (validation_result == ValidationResult::EXPIRED || 
        validation_result == ValidationResult::INSUFFICIENT_DATA) {
        return WeeklyCacheStatus::STALE;
    }
    
    // Check if cache covers current week
    if (!cache_validator_->meetsLastWeekRequirement(cache_path)) {
        return WeeklyCacheStatus::STALE;
    }
    
    // Check data quality
    auto quality = cache_validator_->analyzeCacheQuality(cache_path);
    auto requirement = getWeeklyCacheRequirement();
    
    if (quality.completeness_score < requirement.min_data_quality) {
        return WeeklyCacheStatus::PARTIAL;
    }
    
    return WeeklyCacheStatus::CURRENT;
}

CacheOperationResult WeeklyCacheManager::performCacheBuild(const std::string& pair_symbol, bool force_rebuild) {
    return buildWeeklyCache(pair_symbol, force_rebuild);
}

CacheOperationResult WeeklyCacheManager::performIncrementalUpdate(const std::string& pair_symbol) {
    // For incremental updates, we fetch only the latest data and merge
    // This is a simplified implementation
    return buildWeeklyCache(pair_symbol, false);
}

bool WeeklyCacheManager::mergeCacheData(const std::string& existing_cache, const std::vector<std::string>& new_data) {
    // Simplified merge implementation
    // In a real system, this would intelligently merge new data with existing data
    return true;
}

void WeeklyCacheManager::processQueueEntry(const QueueEntry& entry) {
    auto result = buildWeeklyCache(entry.pair_symbol, false);
    
    if (!result.success) {
        notifyError(entry.pair_symbol, result.message);
    }
}

void WeeklyCacheManager::sortUpdateQueue() {
    std::sort(update_queue_.begin(), update_queue_.end());
}

WeeklyCacheManager::QueueEntry* WeeklyCacheManager::findQueueEntry(const std::string& pair_symbol) {
    auto it = std::find_if(update_queue_.begin(), update_queue_.end(),
                          [&pair_symbol](const QueueEntry& entry) {
                              return entry.pair_symbol == pair_symbol;
                          });
    return (it != update_queue_.end()) ? &(*it) : nullptr;
}

std::string WeeklyCacheManager::getWeeklyCachePath(const std::string& pair_symbol) const {
    return cache_directory_ + pair_symbol + "_weekly.json";
}

std::string WeeklyCacheManager::getBackupCachePath(const std::string& pair_symbol) const {
    return cache_directory_ + pair_symbol + "_weekly.json.backup";
}

bool WeeklyCacheManager::createCacheDirectory() const {
    try {
        fs::create_directories(cache_directory_);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to create cache directory {}: {}", cache_directory_, e.what());
        return false;
    }
}

bool WeeklyCacheManager::backupExistingCache(const std::string& pair_symbol) const {
    std::string cache_path = getWeeklyCachePath(pair_symbol);
    std::string backup_path = getBackupCachePath(pair_symbol);
    
    try {
        if (fs::exists(cache_path)) {
            fs::copy_file(cache_path, backup_path, fs::copy_options::overwrite_existing);
            return true;
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to backup cache for {}: {}", pair_symbol, e.what());
    }
    
    return false;
}

bool WeeklyCacheManager::restoreFromBackup(const std::string& pair_symbol) const {
    std::string cache_path = getWeeklyCachePath(pair_symbol);
    std::string backup_path = getBackupCachePath(pair_symbol);
    
    try {
        if (fs::exists(backup_path)) {
            fs::copy_file(backup_path, cache_path, fs::copy_options::overwrite_existing);
            spdlog::info("Restored cache for {} from backup", pair_symbol);
            return true;
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to restore cache from backup for {}: {}", pair_symbol, e.what());
    }
    
    return false;
}

std::vector<std::string> WeeklyCacheManager::fetchDataForWeek(const std::string& pair_symbol, 
                                                             std::chrono::system_clock::time_point week_start) const {
    std::lock_guard<std::mutex> lock(data_source_mutex_);
    
    if (!data_source_provider_) {
        spdlog::warn("No data source provider configured for fetching weekly data");
        return {};
    }
    
    auto week_end = addDays(week_start, 7);
    return data_source_provider_(pair_symbol, week_start, week_end);
}

bool WeeklyCacheManager::writeCacheFile(const std::string& cache_path, const std::vector<std::string>& data) const {
    try {
        std::ofstream file(cache_path);
        if (!file.is_open()) {
            return false;
        }
        
        // Write data as JSON
        nlohmann::json root;
        nlohmann::json data_array = nlohmann::json::array();
        
        for (const auto& record : data) {
            try {
                // Try to parse as JSON first
                nlohmann::json item = nlohmann::json::parse(record);
                data_array.push_back(item);
            } catch (const nlohmann::json::parse_error&) {
                // If not JSON, store as string
                data_array.push_back(record);
            }
        }
        
        root["data"] = data_array;
        root["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        file << root.dump(2);  // Pretty print with 2-space indentation
        
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to write cache file {}: {}", cache_path, e.what());
        return false;
    }
}

std::vector<std::string> WeeklyCacheManager::readCacheFile(const std::string& cache_path) const {
    std::vector<std::string> data;
    
    try {
        std::ifstream file(cache_path);
        if (!file.is_open()) {
            return data;
        }
        
        // Parse JSON and extract data
        nlohmann::json root;
        file >> root;
        
        if (root.contains("data") && root["data"].is_array()) {
            for (const auto& item : root["data"]) {
                if (item.is_string()) {
                    data.push_back(item.get<std::string>());
                } else {
                    // Convert non-string items back to JSON string
                    data.push_back(item.dump());
                }
            }
        }
        
    } catch (const std::exception& e) {
        spdlog::error("Failed to read cache file {}: {}", cache_path, e.what());
    }
    
    return data;
}

void WeeklyCacheManager::notifyUpdateStatus(const std::string& pair, WeeklyCacheStatus old_status, WeeklyCacheStatus new_status) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : update_callbacks_) {
        try {
            callback(pair, old_status, new_status);
        } catch (const std::exception& e) {
            spdlog::error("Update callback error: {}", e.what());
        }
    }
}

void WeeklyCacheManager::notifyBuildResult(const std::string& pair, const CacheOperationResult& result) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : build_callbacks_) {
        try {
            callback(pair, result);
        } catch (const std::exception& e) {
            spdlog::error("Build callback error: {}", e.what());
        }
    }
}

void WeeklyCacheManager::notifyError(const std::string& pair, const std::string& error_message) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : error_callbacks_) {
        try {
            callback(pair, error_message);
        } catch (const std::exception& e) {
            spdlog::error("Error callback error: {}", e.what());
        }
    }
}

bool WeeklyCacheManager::isWeekendDay(std::chrono::system_clock::time_point timestamp) const {
    std::time_t time = std::chrono::system_clock::to_time_t(timestamp);
    std::tm* tm = std::localtime(&time);
    int weekday = tm->tm_wday; // 0 = Sunday, 6 = Saturday
    return weekday == 0 || weekday == 6;
}

std::chrono::system_clock::time_point WeeklyCacheManager::getWeekStart(std::chrono::system_clock::time_point timestamp) const {
    std::time_t time = std::chrono::system_clock::to_time_t(timestamp);
    std::tm* tm = std::localtime(&time);
    
    // Calculate days since Monday (0 = Monday, 6 = Sunday)
    int days_since_monday = (tm->tm_wday + 6) % 7;
    
    // Go back to Monday at 00:00:00
    tm->tm_mday -= days_since_monday;
    tm->tm_hour = 0;
    tm->tm_min = 0;
    tm->tm_sec = 0;
    
    return std::chrono::system_clock::from_time_t(std::mktime(tm));
}

std::chrono::system_clock::time_point WeeklyCacheManager::addDays(std::chrono::system_clock::time_point timestamp, int days) const {
    return timestamp + std::chrono::hours(24 * days);
}

UpdatePriority WeeklyCacheManager::stringToPriority(const std::string& priority_str) const {
    if (priority_str == "LOW") return UpdatePriority::LOW;
    if (priority_str == "NORMAL") return UpdatePriority::NORMAL;
    if (priority_str == "HIGH") return UpdatePriority::HIGH;
    if (priority_str == "CRITICAL") return UpdatePriority::CRITICAL;
    if (priority_str == "EMERGENCY") return UpdatePriority::EMERGENCY;
    return UpdatePriority::NORMAL;
}

std::string WeeklyCacheManager::priorityToString(UpdatePriority priority) const {
    switch (priority) {
        case UpdatePriority::LOW: return "LOW";
        case UpdatePriority::NORMAL: return "NORMAL";
        case UpdatePriority::HIGH: return "HIGH";
        case UpdatePriority::CRITICAL: return "CRITICAL";
        case UpdatePriority::EMERGENCY: return "EMERGENCY";
        default: return "NORMAL";
    }
}

// Utility functions implementation

std::string weeklyCacheStatusToString(WeeklyCacheStatus status) {
    switch (status) {
        case WeeklyCacheStatus::CURRENT: return "CURRENT";
        case WeeklyCacheStatus::STALE: return "STALE";
        case WeeklyCacheStatus::PARTIAL: return "PARTIAL";
        case WeeklyCacheStatus::MISSING: return "MISSING";
        case WeeklyCacheStatus::BUILDING: return "BUILDING";
        case WeeklyCacheStatus::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

WeeklyCacheStatus stringToWeeklyCacheStatus(const std::string& status_str) {
    if (status_str == "CURRENT") return WeeklyCacheStatus::CURRENT;
    if (status_str == "STALE") return WeeklyCacheStatus::STALE;
    if (status_str == "PARTIAL") return WeeklyCacheStatus::PARTIAL;
    if (status_str == "MISSING") return WeeklyCacheStatus::MISSING;
    if (status_str == "BUILDING") return WeeklyCacheStatus::BUILDING;
    if (status_str == "ERROR") return WeeklyCacheStatus::ERROR;
    return WeeklyCacheStatus::ERROR; // Default for unknown
}

bool isWeeklyCacheReady(WeeklyCacheStatus status) {
    return status == WeeklyCacheStatus::CURRENT;
}

UpdatePriority calculateUpdatePriority(const std::string& pair_symbol, WeeklyCacheStatus status) {
    switch (status) {
        case WeeklyCacheStatus::MISSING:
            return UpdatePriority::HIGH;
        case WeeklyCacheStatus::ERROR:
            return UpdatePriority::CRITICAL;
        case WeeklyCacheStatus::STALE:
            return UpdatePriority::NORMAL;
        case WeeklyCacheStatus::PARTIAL:
            return UpdatePriority::NORMAL;
        case WeeklyCacheStatus::CURRENT:
            return UpdatePriority::LOW;
        case WeeklyCacheStatus::BUILDING:
            return UpdatePriority::LOW;
        default:
            return UpdatePriority::NORMAL;
    }
}

// Global instance
WeeklyCacheManager& getGlobalWeeklyCacheManager() {
    static WeeklyCacheManager instance;
    return instance;
}

} // namespace sep::cache
