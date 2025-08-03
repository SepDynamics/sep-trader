#pragma once

#include <vector>
#include <string>
#include <future>
#include <thread>
#include <atomic>
#include <memory>
#include <functional>

namespace sep::engine::batch {

/**
 * @brief Result of processing a single pattern in a batch
 */
struct BatchResult {
    std::string pattern_id;
    bool success;
    double value;
    std::string error_message;
    
    BatchResult(const std::string& id, bool s, double v, const std::string& err = "")
        : pattern_id(id), success(s), value(v), error_message(err) {}
};

/**
 * @brief Configuration for batch processing
 */
struct BatchConfig {
    size_t max_parallel_threads = std::thread::hardware_concurrency();
    size_t batch_size = 100;
    bool fail_fast = false;  // Stop on first error
    double timeout_seconds = 30.0;
    
    BatchConfig() {
        if (max_parallel_threads == 0) {
            max_parallel_threads = 4;  // Fallback
        }
    }
};

/**
 * @brief Pattern to be processed in batch
 */
struct BatchPattern {
    std::string pattern_id;
    std::string pattern_code;
    std::vector<std::pair<std::string, double>> inputs;  // Variable bindings
    
    BatchPattern(const std::string& id, const std::string& code)
        : pattern_id(id), pattern_code(code) {}
        
    void add_input(const std::string& name, double value) {
        inputs.emplace_back(name, value);
    }
};

/**
 * @brief High-performance batch processor for DSL patterns
 */
class BatchProcessor {
public:
    BatchProcessor(const BatchConfig& config = BatchConfig());
    ~BatchProcessor();
    
    /**
     * @brief Process multiple patterns in parallel
     * @param patterns Vector of patterns to process
     * @return Vector of results (same order as input)
     */
    std::vector<BatchResult> process_batch(const std::vector<BatchPattern>& patterns);
    
    /**
     * @brief Process patterns asynchronously (returns immediately)
     * @param patterns Vector of patterns to process
     * @param callback Function called for each completed pattern
     * @return Future that resolves when all patterns are processed
     */
    std::future<std::vector<BatchResult>> process_batch_async(
        const std::vector<BatchPattern>& patterns,
        std::function<void(const BatchResult&)> callback = nullptr
    );
    
    /**
     * @brief Cancel any running batch operations
     */
    void cancel_batch();
    
    /**
     * @brief Get current batch processing statistics
     */
    struct BatchStats {
        size_t patterns_processed = 0;
        size_t patterns_succeeded = 0;
        size_t patterns_failed = 0;
        double average_processing_time_ms = 0.0;
        double total_processing_time_ms = 0.0;
    };
    
    BatchStats get_batch_stats() const;
    void reset_batch_stats();
    
    /**
     * @brief Update batch configuration
     */
    void update_config(const BatchConfig& config);
    BatchConfig get_config() const;

private:
    BatchConfig config_;
    mutable std::mutex stats_mutex_;
    BatchStats stats_;
    std::atomic<bool> cancel_requested_{false};
    
    /**
     * @brief Process a single pattern (thread-safe)
     */
    BatchResult process_single_pattern(const BatchPattern& pattern);
    
    /**
     * @brief Update statistics for a completed pattern
     */
    void update_stats(bool success, double processing_time_ms);
};

} // namespace sep::engine::batch
