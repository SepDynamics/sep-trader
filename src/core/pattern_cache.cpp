#include <algorithm>
#include <iostream>
#include <sstream>

#include "pattern_cache.h"
#include "core/result_types.h"

namespace sep::engine::cache {

PatternCache::PatternCache(const PatternCacheConfig& config) 
    : config_(config) {
    std::cout << "PatternCache initialized with max_size=" << config_.max_cache_size 
              << ", ttl=" << config_.ttl.count() << "min" << std::endl;
}

sep::Result<void> PatternCache::storePattern(const std::string& key,
                                             const sep::quantum::Pattern& pattern,
                                             float computation_time_ms) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Check if pattern meets caching criteria
    if (!shouldCache(pattern)) {
        return sep::makeError(sep::Error::Code::InvalidArgument, "Pattern does not meet caching criteria");
    }
    
    // Evict expired entries before adding new ones
    evictExpired();
    
    // Check if cache is full and needs LRU eviction
    if (cache_.size() >= config_.max_cache_size && config_.enable_lru_eviction) {
        evictLRU();
    }
    
    // Store the pattern with metadata
    auto entry = std::make_unique<CachedPatternEntry>(pattern, computation_time_ms);
    cache_[key] = std::move(entry);
    
    std::cout << "PatternCache: Stored pattern '" << key << "' (coherence="
              << pattern.coherence << ", computation_time=" << computation_time_ms << "ms)" << std::endl;
    
    return sep::makeSuccess();  // Success case for void
}

sep::Result<bool> PatternCache::retrievePattern(const std::string& key,
                                                sep::quantum::Pattern& pattern) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        cache_misses_++;
        return sep::Result<bool>(false);  // Pattern not found
    }
    
    auto& entry = it->second;
    
    // Check if entry has expired
    auto now = std::chrono::system_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::minutes>(now - entry->created_at);
    if (age > config_.ttl) {
        cache_.erase(it);
        expired_entries_++;
        cache_misses_++;
        return sep::Result<bool>(false);
    }
    
    // Update access metadata
    entry->last_accessed = now;
    entry->access_count++;
    
    // Return cached pattern
    pattern = entry->pattern;
    cache_hits_++;
    
    std::cout << "PatternCache: Retrieved pattern '" << key << "' (cache hit, age="
              << age.count() << "min, access_count=" << entry->access_count << ")" << std::endl;

    return sep::Result<bool>(true);
}

bool PatternCache::hasPattern(const std::string& key) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.find(key) != cache_.end();
}

void PatternCache::clearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    size_t cleared_count = cache_.size();
    cache_.clear();
    std::cout << "PatternCache: Cleared " << cleared_count << " entries" << std::endl;
}

void PatternCache::evictExpired() {
    // Note: Assumes caller holds lock
    auto now = std::chrono::system_clock::now();
    
    for (auto it = cache_.begin(); it != cache_.end();) {
        auto age = std::chrono::duration_cast<std::chrono::minutes>(now - it->second->created_at);
        if (age > config_.ttl) {
            std::cout << "PatternCache: Evicting expired pattern '" << it->first 
                      << "' (age=" << age.count() << "min)" << std::endl;
            it = cache_.erase(it);
            expired_entries_++;
        } else {
            ++it;
        }
    }
}

core::Result<void> PatternCache::configure(const PatternCacheConfig& config) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    config_ = config;
    
    // If new max size is smaller, evict excess entries
    while (cache_.size() > config_.max_cache_size && config_.enable_lru_eviction) {
        evictLRU();
    }
    
    std::cout << "PatternCache: Reconfigured with max_size=" << config_.max_cache_size 
              << ", ttl=" << config_.ttl.count() << "min" << std::endl;
    
    return core::Result<void>();
}

PatternCache::CacheMetrics PatternCache::getMetrics() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    CacheMetrics metrics;
    metrics.total_entries = cache_.size();
    metrics.cache_hits = cache_hits_;
    metrics.cache_misses = cache_misses_;
    metrics.expired_entries = expired_entries_;
    metrics.evicted_entries = evicted_entries_;
    
    // Calculate hit ratio
    size_t total_requests = cache_hits_ + cache_misses_;
    if (total_requests > 0) {
        metrics.hit_ratio = static_cast<float>(cache_hits_) / total_requests;
    }
    
    // Calculate average computation time
    if (!cache_.empty()) {
        float total_time = 0.0f;
        for (const auto& [key, entry] : cache_) {
            total_time += entry->computation_time_ms;
        }
        metrics.average_computation_time_ms = total_time / cache_.size();
    }
    
    return metrics;
}

void PatternCache::resetMetrics() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_hits_ = 0;
    cache_misses_ = 0;
    expired_entries_ = 0;
    evicted_entries_ = 0;
    std::cout << "PatternCache: Metrics reset" << std::endl;
}

void PatternCache::evictLRU() {
    // Note: Assumes caller holds lock
    if (cache_.empty()) return;
    
    // Find least recently used entry
    auto lru_it = std::min_element(cache_.begin(), cache_.end(),
        [](const auto& a, const auto& b) {
            return a.second->last_accessed < b.second->last_accessed;
        });
    
    std::cout << "PatternCache: LRU evicting pattern '" << lru_it->first 
              << "' (last_accessed=" << lru_it->second->access_count << " times)" << std::endl;
    
    cache_.erase(lru_it);
    evicted_entries_++;
}

bool PatternCache::shouldCache(const quantum::Pattern& pattern) const {
    // Only cache patterns with sufficient coherence
    return pattern.coherence >= config_.coherence_cache_threshold;
}

std::string PatternCache::generateCacheKey(const quantum::Pattern& pattern) const {
    std::ostringstream oss;
    oss << pattern.id << "_" << pattern.coherence << "_" << pattern.quantum_state.coherence;
    return oss.str();
}

} // namespace sep::engine::cache
