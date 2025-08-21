#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <optional>

namespace sep::cache {

enum class ValidationLevel {
    QUICK,      // Fast superficial check (integrity/existence)
    STANDARD,   // Standard validation (data format, checksums)
    THOROUGH,   // Deep validation (cross-references, data quality)
    EXHAUSTIVE  // Complete verification (exhaustive data quality analysis)
};

enum class ValidationResult {
    VALID,      // Cache is valid and can be used
    REPAIRABLE, // Cache has issues but can be repaired
    INVALID,    // Cache is invalid and should be rebuilt
    ERROR       // Validation process encountered an error
};

// Add CacheQuality structure that was missing
struct CacheQuality {
    double data_freshness{1.0};    // How recent the data is (0.0-1.0)
    double data_completeness{1.0}; // How complete the coverage is (0.0-1.0)
    double data_consistency{1.0};  // How consistent the data format is (0.0-1.0)
    double anomaly_score{0.0};     // Level of anomalies detected (0.0-1.0, lower is better)
    std::chrono::system_clock::time_point oldest_timestamp;
    std::chrono::system_clock::time_point newest_timestamp;
    size_t record_count{0};
    size_t gap_count{0};
    std::vector<std::string> quality_warnings;
    
    // Add fields expected by CacheHealthMonitor
    double freshness_score{1.0};    // Alias for data_freshness for backward compatibility
    double completeness_score{1.0}; // Alias for data_completeness for backward compatibility
    double consistency_score{1.0};  // Alias for data_consistency for backward compatibility
    
    CacheQuality() = default;
};

struct ValidationMetrics {
    double data_integrity_score{1.0}; // 0.0-1.0 score of data integrity
    double data_quality_score{1.0};   // 0.0-1.0 score of data quality
    double coverage_score{1.0};       // 0.0-1.0 score of temporal coverage
    std::chrono::system_clock::time_point oldest_data_point;
    std::chrono::system_clock::time_point newest_data_point;
    size_t total_records{0};
    size_t validated_records{0};
    size_t invalid_records{0};
    std::vector<std::string> validation_warnings;
    std::vector<std::string> validation_errors;
    
    // Cache-specific metrics that the implementation expects
    size_t cache_size_bytes{0};           // Size of cache file in bytes
    std::time_t last_validation_time{0};  // Last validation timestamp
    bool validation_success{false};        // Overall validation success status
    
    ValidationMetrics() = default;
};

struct ValidationOptions {
    ValidationLevel level{ValidationLevel::STANDARD};
    bool repair_if_possible{true};
    std::chrono::seconds timeout{300}; // 5 minutes default timeout
    bool verbose_logging{false};
    std::optional<std::function<bool(double)>> progress_callback;
    
    ValidationOptions() = default;
};

struct ValidationResponse {
    ValidationResult result;
    ValidationMetrics metrics;
    std::string message;
    bool was_repaired{false};
    std::chrono::duration<double> validation_time;
    std::string repair_details;
    
    ValidationResponse() : result(ValidationResult::ERROR), 
                        validation_time(std::chrono::duration<double>::zero()) {}
};

class CacheValidator {
public:
    CacheValidator();
    ~CacheValidator();
    
    // Core validation methods
    ValidationResponse validateCache(const std::string& cache_name, 
                                  ValidationOptions options = ValidationOptions());
    
    ValidationResponse validateCacheFile(const std::string& file_path,
                                      ValidationOptions options = ValidationOptions());
    
    std::vector<ValidationResponse> validateAllCaches(
        ValidationOptions options = ValidationOptions());
    
    // Utility methods
    bool isCacheValid(const std::string& cache_name);
    bool repairCache(const std::string& cache_name);
    ValidationMetrics getLastValidationMetrics(const std::string& cache_name);
    std::vector<std::string> listValidCaches();
    std::vector<std::string> listInvalidCaches();
    
    // Additional methods needed by CacheHealthMonitor
    std::string getCachePathForPair(const std::string& pair_symbol) const;
    CacheQuality analyzeCacheQuality(const std::string& cache_path) const;
    bool cacheExistsForPair(const std::string& pair_symbol) const;
    
    // Method needed by tests - validates entry sources
    bool validateEntrySources(const std::string& cache_path) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Helper functions
std::string validationResultToString(ValidationResult result);
ValidationResult stringToValidationResult(const std::string& result_str);
std::string validationLevelToString(ValidationLevel level);
ValidationLevel stringToValidationLevel(const std::string& level_str);
bool isValidationSuccessful(ValidationResult result);

} // namespace sep::cache