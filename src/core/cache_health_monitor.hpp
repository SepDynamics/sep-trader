#pragma once

#include <string>

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
    
    // Additional fields expected by trader_cli.cpp
    size_t total_entries{0};    // Total cache entries across all caches
    size_t error_count{0};      // Number of cache errors detected
    
    // Fields expected by trader_cli_simple.cpp
    bool is_healthy{false};           // Overall health status
    size_t cache_size{0};             // Total cache size/entries
    double memory_usage{0.0};         // Memory usage (alias for memory_usage_mb)
    std::string issues;               // Description of any issues
    
    SystemCacheHealth() = default;
};

class CacheHealthMonitor {
public:
    CacheHealthMonitor() = default;
    ~CacheHealthMonitor() = default;
    
    // Method needed by trader CLI
    SystemCacheHealth getHealthStatus() {
        SystemCacheHealth health;
        // Return mock data for now - will be implemented later
        health.is_connected = true;
        health.memory_usage_mb = 128.5;
        health.hit_rate = 0.85;
        health.overall_status = HealthStatus::GOOD;
        health.total_caches = 10;
        health.healthy_caches = 8;
        health.warning_caches = 2;
        health.critical_caches = 0;
        health.failed_caches = 0;
        health.average_health_score = 0.85;
        health.system_performance_score = 0.90;
        return health;
    }
};

} // namespace sep::cache