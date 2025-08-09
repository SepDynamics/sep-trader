#include "nlohmann_json_safe.h"
#include "cache_validator.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <thread>


namespace sep::cache {

namespace fs = std::filesystem;

CacheValidator::CacheValidator() : cache_base_path_("cache/") {
    spdlog::info("CacheValidator initialized with base path: {}", cache_base_path_);
}

ValidationResult CacheValidator::validateCache(const std::string& cache_path) const {
    std::lock_guard<std::mutex> lock(validation_mutex_);
    auto start_time = std::chrono::steady_clock::now();
    
    ValidationResult result = performValidation(cache_path, default_policy_);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time);
    total_validation_time_.store(total_validation_time_.load() + duration);
    total_validations_++;
    
    if (result == ValidationResult::VALID) {
        valid_caches_++;
    } else {
        invalid_caches_++;
    }
    
    CacheQuality quality = analyzeCacheQuality(cache_path);
    notifyValidationResult(cache_path, result, quality);
    
    spdlog::debug("Cache validation completed for {} in {:.3f}ms: {}", 
                  cache_path, duration.count() * 1000, validationResultToString(result));
    
    return result;
}

ValidationResult CacheValidator::validateCacheForPair(const std::string& pair_symbol) const {
    std::string cache_path = getCachePathForPair(pair_symbol);
    
    // Use custom policy if available
    ValidationPolicy policy = default_policy_;
    auto it = pair_policies_.find(pair_symbol);
    if (it != pair_policies_.end()) {
        policy = it->second;
    }
    
    return performValidation(cache_path, policy);
}

CacheQuality CacheValidator::analyzeCacheQuality(const std::string& cache_path) const {
    return analyzeQuality(cache_path, default_policy_);
}

std::unordered_map<std::string, ValidationResult> CacheValidator::validateAllCaches() const {
    std::unordered_map<std::string, ValidationResult> results;
    
    if (!fs::exists(cache_base_path_)) {
        spdlog::warn("Cache base path does not exist: {}", cache_base_path_);
        return results;
    }
    
    for (const auto& entry : fs::directory_iterator(cache_base_path_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            std::string cache_path = entry.path().string();
            results[cache_path] = validateCache(cache_path);
        }
    }
    
    return results;
}

std::unordered_map<std::string, ValidationResult> CacheValidator::validatePairCaches(const std::vector<std::string>& pairs) const {
    std::unordered_map<std::string, ValidationResult> results;
    
    for (const auto& pair : pairs) {
        results[pair] = validateCacheForPair(pair);
    }
    
    return results;
}

void CacheValidator::setValidationPolicy(const ValidationPolicy& policy) {
    std::lock_guard<std::mutex> lock(validation_mutex_);
    default_policy_ = policy;
    spdlog::info("Updated default validation policy");
}

ValidationPolicy CacheValidator::getValidationPolicy() const {
    std::lock_guard<std::mutex> lock(validation_mutex_);
    return default_policy_;
}

void CacheValidator::setCustomPolicyForPair(const std::string& pair, const ValidationPolicy& policy) {
    std::lock_guard<std::mutex> lock(validation_mutex_);
    pair_policies_[pair] = policy;
    spdlog::info("Set custom validation policy for pair: {}", pair);
}

bool CacheValidator::hasCustomPolicy(const std::string& pair) const {
    std::lock_guard<std::mutex> lock(validation_mutex_);
    return pair_policies_.find(pair) != pair_policies_.end();
}

bool CacheValidator::meetsLastWeekRequirement(const std::string& cache_path) const {
    auto now = std::chrono::system_clock::now();
    auto one_week_ago = now - std::chrono::hours(168);
    
    std::vector<std::chrono::system_clock::time_point> timestamps = extractTimestamps(cache_path);
    
    return std::any_of(timestamps.begin(), timestamps.end(),
                      [one_week_ago](const auto& timestamp) {
                          return timestamp >= one_week_ago;
                      });
}

bool CacheValidator::hasMinimumDataCoverage(const std::string& cache_path, std::chrono::hours required_coverage) const {
    std::vector<std::chrono::system_clock::time_point> timestamps = extractTimestamps(cache_path);
    
    if (timestamps.empty()) return false;
    
    auto min_time = *std::min_element(timestamps.begin(), timestamps.end());
    auto max_time = *std::max_element(timestamps.begin(), timestamps.end());
    
    auto coverage = std::chrono::duration_cast<std::chrono::hours>(max_time - min_time);
    return coverage >= required_coverage;
}

bool CacheValidator::isDataContinuous(const std::string& cache_path) const {
    std::vector<std::chrono::system_clock::time_point> timestamps = extractTimestamps(cache_path);
    
    if (timestamps.size() < 2) return false;
    
    std::sort(timestamps.begin(), timestamps.end());
    
    auto gaps = findDataGaps(timestamps);
    auto max_allowed_gap = default_policy_.max_data_gap;
    
    return std::all_of(gaps.begin(), gaps.end(),
                      [max_allowed_gap](const auto& gap) {
                          return gap <= max_allowed_gap;
                      });
}

void CacheValidator::enableContinuousMonitoring(bool enable) {
    continuous_monitoring_enabled_ = enable;
    
    if (enable && !monitoring_thread_) {
        stop_monitoring_ = false;
        monitoring_thread_ = std::make_unique<std::thread>(&CacheValidator::monitoringLoop, this);
        spdlog::info("Continuous cache monitoring enabled");
    } else if (!enable && monitoring_thread_) {
        stop_monitoring_ = true;
        if (monitoring_thread_->joinable()) {
            monitoring_thread_->join();
        }
        monitoring_thread_.reset();
        spdlog::info("Continuous cache monitoring disabled");
    }
}

bool CacheValidator::isContinuousMonitoringEnabled() const {
    return continuous_monitoring_enabled_;
}

void CacheValidator::setMonitoringInterval(std::chrono::minutes interval) {
    monitoring_interval_ = interval;
    spdlog::info("Cache monitoring interval set to {} minutes", interval.count());
}

std::chrono::minutes CacheValidator::getMonitoringInterval() const {
    return monitoring_interval_;
}

size_t CacheValidator::addValidationCallback(ValidationCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    validation_callbacks_.push_back(std::move(callback));
    return validation_callbacks_.size() - 1;
}

void CacheValidator::removeValidationCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < validation_callbacks_.size()) {
        validation_callbacks_.erase(validation_callbacks_.begin() + callback_id);
    }
}

size_t CacheValidator::addRepairCallback(RepairCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    repair_callbacks_.push_back(std::move(callback));
    return repair_callbacks_.size() - 1;
}

void CacheValidator::removeRepairCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < repair_callbacks_.size()) {
        repair_callbacks_.erase(repair_callbacks_.begin() + callback_id);
    }
}

bool CacheValidator::attemptCacheRepair(const std::string& cache_path) {
    spdlog::info("Attempting to repair cache: {}", cache_path);
    
    auto suggestions = suggestRepairActions(cache_path);
    return requestRepair(cache_path, suggestions);
}

bool CacheValidator::rebuildCacheFromSource(const std::string& cache_path, const std::string& pair_symbol) {
    spdlog::info("Rebuilding cache from source for pair: {}", pair_symbol);
    
    // This would integrate with data source providers
    // For now, return false as this requires external data source integration
    spdlog::warn("Cache rebuild requires external data source integration (not implemented)");
    return false;
}

std::vector<std::string> CacheValidator::suggestRepairActions(const std::string& cache_path) const {
    std::vector<std::string> suggestions;
    
    if (!isFileReadable(cache_path)) {
        suggestions.push_back("Check file permissions");
        suggestions.push_back("Verify file exists and is accessible");
    }
    
    if (!validateJsonStructure(cache_path)) {
        suggestions.push_back("Repair JSON structure");
        suggestions.push_back("Restore from backup if available");
    }
    
    if (!meetsLastWeekRequirement(cache_path)) {
        suggestions.push_back("Fetch recent data");
        suggestions.push_back("Update cache with current data");
    }
    
    return suggestions;
}

size_t CacheValidator::getTotalCachesValidated() const {
    return total_validations_;
}

size_t CacheValidator::getValidCacheCount() const {
    return valid_caches_;
}

size_t CacheValidator::getInvalidCacheCount() const {
    return invalid_caches_;
}

double CacheValidator::getOverallCacheHealthScore() const {
    size_t total = valid_caches_ + invalid_caches_;
    if (total == 0) return 0.0;
    return static_cast<double>(valid_caches_) / total;
}

void CacheValidator::setCacheBasePath(const std::string& base_path) {
    std::lock_guard<std::mutex> lock(validation_mutex_);
    cache_base_path_ = base_path;
    if (!cache_base_path_.empty() && cache_base_path_.back() != '/') {
        cache_base_path_ += '/';
    }
    spdlog::info("Cache base path set to: {}", cache_base_path_);
}

std::string CacheValidator::getCacheBasePath() const {
    std::lock_guard<std::mutex> lock(validation_mutex_);
    return cache_base_path_;
}

std::string CacheValidator::getCachePathForPair(const std::string& pair_symbol) const {
    return cache_base_path_ + pair_symbol + "_cache.json";
}

bool CacheValidator::cacheExistsForPair(const std::string& pair_symbol) const {
    std::string cache_path = getCachePathForPair(pair_symbol);
    return fs::exists(cache_path);
}

bool CacheValidator::validateCacheStructure(const std::string& cache_path) const {
    return validateJsonStructure(cache_path);
}

bool CacheValidator::validateDataIntegrity(const std::string& cache_path) const {
    // Basic integrity check - ensure file is readable and has valid structure
    return isFileReadable(cache_path) && validateJsonStructure(cache_path);
}

bool CacheValidator::checkForCorruption(const std::string& cache_path) const {
    return !validateDataIntegrity(cache_path);
}

std::chrono::duration<double> CacheValidator::getLastValidationTime() const {
    // Return average as approximation for last validation time
    return getAverageValidationTime();
}

std::chrono::duration<double> CacheValidator::getAverageValidationTime() const {
    size_t total = total_validations_;
    if (total == 0) return std::chrono::duration<double>::zero();
    return total_validation_time_.load() / total;
}

size_t CacheValidator::getValidationCount() const {
    return total_validations_;
}

// Private methods implementation

ValidationResult CacheValidator::performValidation(const std::string& cache_path, const ValidationPolicy& policy) const {
    if (!checkFileAccessibility(cache_path)) {
        return ValidationResult::MISSING;
    }
    
    if (!validateJsonStructure(cache_path)) {
        return ValidationResult::CORRUPTED;
    }
    
    if (!checkDataAge(cache_path, policy)) {
        return ValidationResult::EXPIRED;
    }
    
    if (!checkDataCompleteness(cache_path, policy)) {
        return ValidationResult::INSUFFICIENT_DATA;
    }
    
    if (!checkDataContinuity(cache_path, policy)) {
        return ValidationResult::INSUFFICIENT_DATA;
    }
    
    return ValidationResult::VALID;
}

CacheQuality CacheValidator::analyzeQuality(const std::string& cache_path, const ValidationPolicy& policy) const {
    CacheQuality quality;
    
    if (!isFileReadable(cache_path)) {
        quality.issues.push_back("File not accessible");
        return quality;
    }
    
    std::vector<std::chrono::system_clock::time_point> timestamps = extractTimestamps(cache_path);
    
    if (!timestamps.empty()) {
        quality.oldest_data = *std::min_element(timestamps.begin(), timestamps.end());
        quality.newest_data = *std::max_element(timestamps.begin(), timestamps.end());
        quality.total_records = timestamps.size();
    }
    
    quality.completeness_score = calculateCompletenessScore(cache_path, policy);
    quality.freshness_score = calculateFreshnessScore(timestamps);
    quality.consistency_score = calculateConsistencyScore(cache_path);
    
    // Calculate missing records estimate
    auto now = std::chrono::system_clock::now();
    auto coverage_period = std::chrono::duration_cast<std::chrono::minutes>(now - quality.oldest_data);
    size_t expected_records = coverage_period.count(); // Assuming 1 record per minute
    quality.missing_records = (expected_records > quality.total_records) ? 
                               expected_records - quality.total_records : 0;
    
    return quality;
}

bool CacheValidator::checkFileAccessibility(const std::string& cache_path) const {
    return fs::exists(cache_path) && isFileReadable(cache_path);
}

bool CacheValidator::checkDataAge(const std::string& cache_path, const ValidationPolicy& policy) const {
    auto file_time = getFileModificationTime(cache_path);
    auto now = std::chrono::system_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::hours>(now - file_time);
    return age <= policy.max_age;
}

bool CacheValidator::checkDataCompleteness(const std::string& cache_path, const ValidationPolicy& policy) const {
    double completeness = calculateCompletenessScore(cache_path, policy);
    return completeness >= policy.min_completeness;
}

bool CacheValidator::checkDataContinuity(const std::string& cache_path, const ValidationPolicy& policy) const {
    if (!policy.require_continuous_data) return true;
    
    std::vector<std::chrono::system_clock::time_point> timestamps = extractTimestamps(cache_path);
    auto gaps = findDataGaps(timestamps);
    
    return std::all_of(gaps.begin(), gaps.end(),
                      [&policy](const auto& gap) {
                          return gap <= policy.max_data_gap;
                      });
}

std::vector<std::chrono::system_clock::time_point> CacheValidator::extractTimestamps(const std::string& cache_path) const {
    std::vector<std::chrono::system_clock::time_point> timestamps;
    
    std::vector<std::chrono::system_clock::time_point> temp_timestamps;
    if (parseJsonCache(cache_path, temp_timestamps)) {
        timestamps = std::move(temp_timestamps);
    }
    
    return timestamps;
}

size_t CacheValidator::countRecords(const std::string& cache_path) const {
    return extractTimestamps(cache_path).size();
}

std::vector<std::chrono::minutes> CacheValidator::findDataGaps(const std::vector<std::chrono::system_clock::time_point>& timestamps) const {
    std::vector<std::chrono::minutes> gaps;
    
    if (timestamps.size() < 2) return gaps;
    
    auto sorted_timestamps = timestamps;
    std::sort(sorted_timestamps.begin(), sorted_timestamps.end());
    
    for (size_t i = 1; i < sorted_timestamps.size(); ++i) {
        auto gap = std::chrono::duration_cast<std::chrono::minutes>(
            sorted_timestamps[i] - sorted_timestamps[i-1]);
        gaps.push_back(gap);
    }
    
    return gaps;
}

double CacheValidator::calculateCompletenessScore(const std::string& cache_path, const ValidationPolicy& policy) const {
    auto timestamps = extractTimestamps(cache_path);
    if (timestamps.empty()) return 0.0;
    
    auto min_time = *std::min_element(timestamps.begin(), timestamps.end());
    auto max_time = *std::max_element(timestamps.begin(), timestamps.end());
    
    auto coverage_minutes = std::chrono::duration_cast<std::chrono::minutes>(max_time - min_time);
    size_t expected_records = coverage_minutes.count(); // Assuming 1 record per minute
    
    if (expected_records == 0) return 1.0;
    
    return std::min(1.0, static_cast<double>(timestamps.size()) / expected_records);
}

double CacheValidator::calculateFreshnessScore(const std::vector<std::chrono::system_clock::time_point>& timestamps) const {
    if (timestamps.empty()) return 0.0;
    
    auto newest = *std::max_element(timestamps.begin(), timestamps.end());
    auto now = std::chrono::system_clock::now();
    auto age_hours = std::chrono::duration_cast<std::chrono::hours>(now - newest);
    
    // Score decreases as data gets older
    double max_acceptable_age_hours = 168.0; // 7 days
    if (age_hours.count() >= max_acceptable_age_hours) return 0.0;
    
    return 1.0 - (age_hours.count() / max_acceptable_age_hours);
}

double CacheValidator::calculateConsistencyScore(const std::string& cache_path) const {
    // Basic consistency check - if we can parse it, it's consistent
    return validateJsonStructure(cache_path) ? 1.0 : 0.0;
}

bool CacheValidator::isValidCacheFile(const std::string& cache_path) const {
    return fs::exists(cache_path) && fs::is_regular_file(cache_path) && 
           (cache_path.length() >= 5 && cache_path.substr(cache_path.length() - 5) == ".json");
}

std::chrono::system_clock::time_point CacheValidator::getFileModificationTime(const std::string& cache_path) const {
    try {
        auto file_time = fs::last_write_time(cache_path);
        auto sys_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            file_time - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
        return sys_time;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to get file modification time for {}: {}", cache_path, e.what());
        return std::chrono::system_clock::now();
    }
}

size_t CacheValidator::getFileSize(const std::string& cache_path) const {
    try {
        return fs::file_size(cache_path);
    } catch (const std::exception& e) {
        spdlog::warn("Failed to get file size for {}: {}", cache_path, e.what());
        return 0;
    }
}

bool CacheValidator::isFileReadable(const std::string& cache_path) const {
    std::ifstream file(cache_path);
    return file.good();
}

void CacheValidator::notifyValidationResult(const std::string& cache_path, ValidationResult result, const CacheQuality& quality) const {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    for (const auto& callback : validation_callbacks_) {
        try {
            callback(cache_path, result, quality);
        } catch (const std::exception& e) {
            spdlog::error("Validation callback error: {}", e.what());
        }
    }
}

bool CacheValidator::requestRepair(const std::string& cache_path, const std::vector<std::string>& issues) const {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : repair_callbacks_) {
        try {
            if (callback(cache_path, issues)) {
                return true;
            }
        } catch (const std::exception& e) {
            spdlog::error("Repair callback error: {}", e.what());
        }
    }
    
    return false;
}

void CacheValidator::monitoringLoop() {
    while (!stop_monitoring_) {
        performPeriodicValidation();
        std::this_thread::sleep_for(monitoring_interval_);
    }
}

void CacheValidator::performPeriodicValidation() {
    auto results = validateAllCaches();
    
    size_t valid_count = 0;
    for (const auto& [path, result] : results) {
        if (result == ValidationResult::VALID) {
            valid_count++;
        }
    }
    
    spdlog::debug("Periodic validation completed: {}/{} caches valid", 
                  valid_count, results.size());
}

bool CacheValidator::parseJsonCache(const std::string& cache_path, std::vector<std::chrono::system_clock::time_point>& timestamps) const {
    try {
        std::ifstream file(cache_path);
        if (!file.is_open()) return false;
        
        nlohmann::json root;
        file >> root;
        
        // Assuming the JSON structure has a "data" array with timestamp fields
        if (root.contains("data") && root["data"].is_array()) {
            for (const auto& entry : root["data"]) {
                if (entry.contains("timestamp")) {
                    if (entry["timestamp"].is_string()) {
                        // Parse timestamp string to time_point
                        // This is a simplified implementation - real timestamp parsing would be more robust
                        std::string timestamp_str = entry["timestamp"].get<std::string>();
                        // For now, use current time as placeholder
                        timestamps.push_back(std::chrono::system_clock::now());
                    } else if (entry["timestamp"].is_number()) {
                        // Unix timestamp
                        auto timestamp = std::chrono::system_clock::from_time_t(entry["timestamp"].get<time_t>());
                        timestamps.push_back(timestamp);
                    }
                }
            }
        }
        
        return !timestamps.empty();
    } catch (const std::exception& e) {
        spdlog::error("Error parsing JSON cache {}: {}", cache_path, e.what());
        return false;
    }
}

bool CacheValidator::validateJsonStructure(const std::string& cache_path) const {
    try {
        std::ifstream file(cache_path);
        if (!file.is_open()) return false;
        
        nlohmann::json root;
        file >> root;
        
        return true; // If we reach here, JSON was parsed successfully
    } catch (const std::exception& e) {
        return false;
    }
}

// Utility functions implementation

std::string validationResultToString(ValidationResult result) {
    switch (result) {
        case ValidationResult::VALID: return "VALID";
        case ValidationResult::EXPIRED: return "EXPIRED";
        case ValidationResult::MISSING: return "MISSING";
        case ValidationResult::CORRUPTED: return "CORRUPTED";
        case ValidationResult::INSUFFICIENT_DATA: return "INSUFFICIENT_DATA";
        case ValidationResult::ACCESS_ERROR: return "ACCESS_ERROR";
        default: return "UNKNOWN";
    }
}

ValidationResult stringToValidationResult(const std::string& result_str) {
    if (result_str == "VALID") return ValidationResult::VALID;
    if (result_str == "EXPIRED") return ValidationResult::EXPIRED;
    if (result_str == "MISSING") return ValidationResult::MISSING;
    if (result_str == "CORRUPTED") return ValidationResult::CORRUPTED;
    if (result_str == "INSUFFICIENT_DATA") return ValidationResult::INSUFFICIENT_DATA;
    if (result_str == "ACCESS_ERROR") return ValidationResult::ACCESS_ERROR;
    return ValidationResult::ACCESS_ERROR; // Default for unknown
}

bool isValidationResultSuccess(ValidationResult result) {
    return result == ValidationResult::VALID;
}

double calculateOverallCacheScore(const CacheQuality& quality) {
    return (quality.completeness_score + quality.freshness_score + quality.consistency_score) / 3.0;
}

// Global instance
CacheValidator& getGlobalCacheValidator() {
    static CacheValidator instance;
    return instance;
}

} // namespace sep::cache
