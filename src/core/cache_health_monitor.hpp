#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include "cache_validator.hpp"
#include "util/nlohmann_json_safe.h"

namespace sep::cache {

enum class HealthStatus {
    EXCELLENT,  // All caches healthy, no issues
    GOOD,       // Minor issues but functional
    WARNING,    // Some issues requiring attention
    CRITICAL,   // Major issues affecting functionality
    FAILURE     // System-wide cache failure
};

// Simplified health status struct for CLI usage
struct SystemCacheHealth {
    HealthStatus overall_status{HealthStatus::FAILURE};
    bool is_connected{false};
    double memory_usage_mb{0.0};
    double hit_rate{0.0};
    size_t total_caches{0};
    size_t healthy_caches{0};
    size_t warning_caches{0};
    size_t critical_caches{0};
    size_t failed_caches{0};
    double average_health_score{0.0};
    double system_performance_score{0.0};

    // Extended fields used by CLI and C interfaces
    size_t total_entries{0};  // Total cache entries across all caches
    size_t error_count{0};    // Number of cache errors detected
    bool is_healthy{false};    // Overall health status
    size_t cache_size{0};      // Total cache size/entries
    double memory_usage{0.0};  // Memory usage (alias for memory_usage_mb)
    std::string issues;        // Description of any issues

    SystemCacheHealth() = default;
};

class CacheHealthMonitor {
  public:
    CacheHealthMonitor() = default;
    ~CacheHealthMonitor() = default;

    // Method needed by trader CLI
    SystemCacheHealth getHealthStatus() {
        SystemCacheHealth health;
        namespace fs = std::filesystem;
        CacheValidator validator;

        health.is_connected = fs::exists("cache");
        if (!health.is_connected) {
            health.overall_status = HealthStatus::FAILURE;
            return health;
        }

        for (const auto& entry : fs::directory_iterator("cache")) {
            if (!entry.is_regular_file() || entry.path().extension() != ".json") {
                continue;
            }

            ++health.total_caches;
            auto size = entry.file_size();
            health.memory_usage_mb += static_cast<double>(size) / (1024.0 * 1024.0);

            auto response = validator.validateCacheFile(entry.path().string());
            switch (response.result) {
                case ValidationResult::VALID:
                    ++health.healthy_caches;
                    break;
                case ValidationResult::REPAIRABLE:
                    ++health.warning_caches;
                    break;
                case ValidationResult::INVALID:
                    ++health.critical_caches;
                    break;
                case ValidationResult::ERROR:
                    ++health.failed_caches;
                    break;
            }

            std::ifstream file(entry.path());
            try {
                nlohmann::json data;
                file >> data;
                if (data.contains("data") && data["data"].is_array()) {
                    health.total_entries += data["data"].size();
                }
            } catch (...) {
                ++health.error_count;
            }
        }

        if (health.total_caches > 0) {
            health.hit_rate =
                health.total_entries > 0
                    ? 1.0 - static_cast<double>(health.error_count) / health.total_entries
                    : 1.0;
            health.average_health_score =
                static_cast<double>(health.healthy_caches) / health.total_caches;
            health.system_performance_score = health.average_health_score;
            health.cache_size = health.total_entries;
            health.memory_usage = health.memory_usage_mb;
            health.is_healthy = (health.warning_caches == 0 && health.critical_caches == 0 &&
                                 health.failed_caches == 0);

            if (health.failed_caches > 0 || health.critical_caches > 0) {
                health.overall_status = HealthStatus::CRITICAL;
            } else if (health.warning_caches > 0) {
                health.overall_status = HealthStatus::WARNING;
            } else {
                health.overall_status = HealthStatus::GOOD;
            }

            if (!health.is_healthy) {
                health.issues = "Cache inconsistencies detected";
            }
        } else {
            health.overall_status = HealthStatus::FAILURE;
        }

        return health;
    }
};

}  // namespace sep::cache
