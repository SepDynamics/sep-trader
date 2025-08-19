#include "core/cache_validator.hpp"
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <vector>
#include <memory>
#include "util/nlohmann_json_safe.h"

namespace sep::cache {

class CacheValidator::Impl {
public:
    bool performValidation(const std::string& cache_name, ValidationOptions options) {
        // Basic validation implementation
        return !cache_name.empty();
    }
};

CacheValidator::CacheValidator() : impl_(std::make_unique<Impl>()) {}
CacheValidator::~CacheValidator() = default;

ValidationResponse CacheValidator::validateCache(const std::string& cache_name,
                                            ValidationOptions options) {
    ValidationResponse response;
    bool is_valid = impl_->performValidation(cache_name, options);
    response.result = is_valid ? ValidationResult::VALID : ValidationResult::INVALID;
    response.message = "Cache validation for: " + cache_name;
    return response;
}

ValidationResponse CacheValidator::validateCacheFile(const std::string& file_path,
                                                ValidationOptions options) {
    ValidationResponse response;
    bool is_valid = !file_path.empty();
    response.result = is_valid ? ValidationResult::VALID : ValidationResult::INVALID;
    response.message = "File validation for: " + file_path;
    return response;
}

std::vector<ValidationResponse> CacheValidator::validateAllCaches(
    ValidationOptions options) {
    return std::vector<ValidationResponse>{};
}

bool CacheValidator::isCacheValid(const std::string& cache_name) {
    return impl_->performValidation(cache_name, ValidationOptions{});
}

bool CacheValidator::repairCache(const std::string& cache_name) {
    return true; // Stub implementation
}

ValidationMetrics CacheValidator::getLastValidationMetrics(const std::string& cache_name) {
    return ValidationMetrics{}; // Default constructed
}

std::vector<std::string> CacheValidator::listValidCaches() {
    return std::vector<std::string>{};
}

std::vector<std::string> CacheValidator::listInvalidCaches() {
    return std::vector<std::string>{};
}

std::string CacheValidator::getCachePathForPair(const std::string& pair_symbol) const {
    return "cache/" + pair_symbol + ".json"; // Basic implementation
}

CacheQuality CacheValidator::analyzeCacheQuality(const std::string& cache_path) const {
    return CacheQuality{}; // Default constructed
}

bool CacheValidator::cacheExistsForPair(const std::string& pair_symbol) const {
    std::string path = getCachePathForPair(pair_symbol);
    std::ifstream file(path);
    return file.good();
}

bool CacheValidator::validateEntrySources(const std::string& cache_path) const {
    try {
        std::ifstream file(cache_path);
        if (!file.is_open()) {
            return false;
        }
        
        nlohmann::json data;
        file >> data;
        
        // Check if data has entries with non-stub providers
        if (data.contains("data") && data["data"].is_array()) {
            for (const auto& entry : data["data"]) {
                if (entry.contains("provider")) {
                    std::string provider = entry["provider"];
                    if (provider == "stub") {
                        return false; // Reject stub providers
                    }
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace sep::cache