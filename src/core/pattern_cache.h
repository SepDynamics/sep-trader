#pragma once

#include "pattern.h"
#include "util/result.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <chrono>

namespace sep::engine::cache {

/// Configuration for pattern caching behavior
struct PatternCacheConfig {
    size_t max_cache_size{1000};           // Maximum number of cached patterns
    std::chrono::minutes ttl{60};          // Time-to-live for cached entries  
    bool enable_lru_eviction{true};        // Use LRU eviction when cache is full
    float coherence_cache_threshold{0.3f}; // Only cache patterns above this coherence
};

/// Cached pattern entry with metadata
struct CachedPatternEntry {
    core::Pattern pattern;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_accessed;
    size_t access_count{0};
    float computation_time_ms{0.0f};       // Original computation time
    
    CachedPatternEntry(const core::Pattern& p, float comp_time = 0.0f)
        : pattern(p), computation_time_ms(comp_time) {
        auto now = std::chrono::system_clock::now();
        created_at = now;
        last_accessed = now;
        access_count = 1;
    }
};

/// High-performance pattern cache with intelligent eviction
class PatternCache {
public:
    explicit PatternCache(const PatternCacheConfig& config = PatternCacheConfig{});
    ~PatternCache() = default;

    // Core cache operations
    core::Result storePattern(const std::string& key, const core::Pattern& pattern, 
                             float computation_time_ms = 0.0f);
    core::Result retrievePattern(const std::string& key, core::Pattern& pattern);
    bool hasPattern(const std::string& key) const;
    
    // Cache management
    void clearCache();
    void evictExpired();
    core::Result configure(const PatternCacheConfig& config);
    
    // Analytics
    struct CacheMetrics {
        size_t total_entries{0};
        size_t cache_hits{0};
        size_t cache_misses{0};
        float hit_ratio{0.0f};
        float average_computation_time_ms{0.0f};
        size_t expired_entries{0};
        size_t evicted_entries{0};
    };
    
    CacheMetrics getMetrics() const;
    void resetMetrics();

    // Thread safety
    PatternCache(const PatternCache&) = delete;
    PatternCache& operator=(const PatternCache&) = delete;
    PatternCache(PatternCache&&) = delete;
    PatternCache& operator=(PatternCache&&) = delete;

private:
    void evictLRU();
    bool shouldCache(const core::Pattern& pattern) const;
    std::string generateCacheKey(const core::Pattern& pattern) const;
    
    mutable std::mutex cache_mutex_;
    PatternCacheConfig config_;
    std::unordered_map<std::string, std::unique_ptr<CachedPatternEntry>> cache_;
    
    // Metrics (thread-safe via mutex)
    mutable size_t cache_hits_{0};
    mutable size_t cache_misses_{0};
    mutable size_t expired_entries_{0};
    mutable size_t evicted_entries_{0};
};

} // namespace sep::engine::cache
