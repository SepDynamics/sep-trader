#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>

namespace sep::cache {

// Cache validation result
enum class ValidationResult {
    VALID,              // Cache is valid and ready
    EXPIRED,            // Cache exists but is expired
    MISSING,            // Cache file does not exist
    CORRUPTED,          // Cache file exists but is corrupted
    INSUFFICIENT_DATA,  // Cache doesn't have enough data
    ACCESS_ERROR        // Permission or I/O error
};

// Cache quality metrics
struct CacheQuality {
    double completeness_score;      // 0.0-1.0, percentage of expected data present
    double freshness_score;         // 0.0-1.0, how recent the data is
    double consistency_score;       // 0.0-1.0, data consistency check
    std::chrono::system_clock::time_point oldest_data;
    std::chrono::system_clock::time_point newest_data;
    size_t total_records;
    size_t missing_records;
    std::vector<std::string> issues;
    
    CacheQuality() : completeness_score(0.0), freshness_score(0.0), 
                    consistency_score(0.0), total_records(0), missing_records(0) {}
};

// Cache validation policy
struct ValidationPolicy {
    std::chrono::hours max_age{168}; // 7 days default
    std::chrono::hours min_coverage{168}; // Must cover at least 7 days
    double min_completeness{0.85}; // Must have 85% of expected data
    double min_freshness{0.7}; // Data must be reasonably fresh
    bool require_continuous_data{true}; // No large gaps allowed
    std::chrono::minutes max_data_gap{60}; // Max 60 minute gaps
    size_t min_records_per_day{1440}; // Minimum records per day (1 per minute)
    
    ValidationPolicy() = default;
};

// Cache validation callback types
using ValidationCallback = std::function<void(const std::string& cache_path, ValidationResult result, const CacheQuality& quality)>;
using RepairCallback = std::function<bool(const std::string& cache_path, const std::vector<std::string>& issues)>;

class CacheValidator {
public:
    CacheValidator();
    ~CacheValidator() = default;

    // Primary validation methods
    ValidationResult validateCache(const std::string& cache_path) const;
    ValidationResult validateCacheForPair(const std::string& pair_symbol) const;
    CacheQuality analyzeCacheQuality(const std::string& cache_path) const;
    
    // Batch validation
    std::unordered_map<std::string, ValidationResult> validateAllCaches() const;
    std::unordered_map<std::string, ValidationResult> validatePairCaches(const std::vector<std::string>& pairs) const;
    
    // Policy management
    void setValidationPolicy(const ValidationPolicy& policy);
    ValidationPolicy getValidationPolicy() const;
    void setCustomPolicyForPair(const std::string& pair, const ValidationPolicy& policy);
    bool hasCustomPolicy(const std::string& pair) const;
    
    // Cache requirements validation
    bool meetsLastWeekRequirement(const std::string& cache_path) const;
    bool hasMinimumDataCoverage(const std::string& cache_path, std::chrono::hours required_coverage) const;
    bool isDataContinuous(const std::string& cache_path) const;
    
    // Cache health monitoring
    void enableContinuousMonitoring(bool enable = true);
    bool isContinuousMonitoringEnabled() const;
    void setMonitoringInterval(std::chrono::minutes interval);
    std::chrono::minutes getMonitoringInterval() const;
    
    // Event system
    size_t addValidationCallback(ValidationCallback callback);
    void removeValidationCallback(size_t callback_id);
    size_t addRepairCallback(RepairCallback callback);
    void removeRepairCallback(size_t callback_id);
    
    // Cache repair and recovery
    bool attemptCacheRepair(const std::string& cache_path);
    bool rebuildCacheFromSource(const std::string& cache_path, const std::string& pair_symbol);
    std::vector<std::string> suggestRepairActions(const std::string& cache_path) const;
    
    // Cache statistics
    size_t getTotalCachesValidated() const;
    size_t getValidCacheCount() const;
    size_t getInvalidCacheCount() const;
    double getOverallCacheHealthScore() const;
    
    // Cache path management
    void setCacheBasePath(const std::string& base_path);
    std::string getCacheBasePath() const;
    std::string getCachePathForPair(const std::string& pair_symbol) const;
    bool cacheExistsForPair(const std::string& pair_symbol) const;
    
    // Data integrity checks
    bool validateCacheStructure(const std::string& cache_path) const;
    bool validateDataIntegrity(const std::string& cache_path) const;
    bool checkForCorruption(const std::string& cache_path) const;
    
    // Performance metrics
    std::chrono::duration<double> getLastValidationTime() const;
    std::chrono::duration<double> getAverageValidationTime() const;
    size_t getValidationCount() const;

private:
    mutable std::mutex validation_mutex_;
    ValidationPolicy default_policy_;
    std::unordered_map<std::string, ValidationPolicy> pair_policies_;
    std::string cache_base_path_;
    
    // Monitoring
    std::atomic<bool> continuous_monitoring_enabled_{false};
    std::chrono::minutes monitoring_interval_{15}; // 15 minutes default
    std::unique_ptr<std::thread> monitoring_thread_;
    std::atomic<bool> stop_monitoring_{false};
    
    // Event callbacks
    std::vector<ValidationCallback> validation_callbacks_;
    std::vector<RepairCallback> repair_callbacks_;
    mutable std::mutex callbacks_mutex_;
    
    // Statistics
    mutable std::atomic<size_t> total_validations_{0};
    mutable std::atomic<size_t> valid_caches_{0};
    mutable std::atomic<size_t> invalid_caches_{0};
    mutable std::atomic<std::chrono::duration<double>> total_validation_time_{std::chrono::duration<double>::zero()};
    
    // Internal validation methods
    ValidationResult performValidation(const std::string& cache_path, const ValidationPolicy& policy) const;
    CacheQuality analyzeQuality(const std::string& cache_path, const ValidationPolicy& policy) const;
    bool checkFileAccessibility(const std::string& cache_path) const;
    bool checkDataAge(const std::string& cache_path, const ValidationPolicy& policy) const;
    bool checkDataCompleteness(const std::string& cache_path, const ValidationPolicy& policy) const;
    bool checkDataContinuity(const std::string& cache_path, const ValidationPolicy& policy) const;
    
    // Data analysis helpers
    std::vector<std::chrono::system_clock::time_point> extractTimestamps(const std::string& cache_path) const;
    size_t countRecords(const std::string& cache_path) const;
    std::vector<std::chrono::minutes> findDataGaps(const std::vector<std::chrono::system_clock::time_point>& timestamps) const;
    double calculateCompletenessScore(const std::string& cache_path, const ValidationPolicy& policy) const;
    double calculateFreshnessScore(const std::vector<std::chrono::system_clock::time_point>& timestamps) const;
    double calculateConsistencyScore(const std::string& cache_path) const;
    
    // File operations
    bool isValidCacheFile(const std::string& cache_path) const;
    std::chrono::system_clock::time_point getFileModificationTime(const std::string& cache_path) const;
    size_t getFileSize(const std::string& cache_path) const;
    bool isFileReadable(const std::string& cache_path) const;
    
    // Event notification
    void notifyValidationResult(const std::string& cache_path, ValidationResult result, const CacheQuality& quality) const;
    bool requestRepair(const std::string& cache_path, const std::vector<std::string>& issues) const;
    
    // Monitoring thread
    void monitoringLoop();
    void performPeriodicValidation();
    
    // JSON parsing helpers (for cache content analysis)
    bool parseJsonCache(const std::string& cache_path, std::vector<std::chrono::system_clock::time_point>& timestamps) const;
    bool validateJsonStructure(const std::string& cache_path) const;
};

// Utility functions
std::string validationResultToString(ValidationResult result);
ValidationResult stringToValidationResult(const std::string& result_str);
bool isValidationResultSuccess(ValidationResult result);
double calculateOverallCacheScore(const CacheQuality& quality);

// Global cache validator instance
CacheValidator& getGlobalCacheValidator();

} // namespace sep::cache
