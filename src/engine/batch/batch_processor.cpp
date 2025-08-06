#include "batch_processor.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>

#include "facade/facade.h"

namespace sep::engine::batch {

BatchProcessor::BatchProcessor(const BatchConfig& config) : config_(config) {
}

BatchProcessor::~BatchProcessor() {
    cancel_batch();
}

std::vector<BatchResult> BatchProcessor::process_batch(const std::vector<BatchPattern>& patterns) {
    if (patterns.empty()) {
        return {};
    }
    
    cancel_requested_ = false;
    std::vector<BatchResult> results;
    results.reserve(patterns.size());
    
    const size_t num_threads = std::min(config_.max_parallel_threads, patterns.size());
    const size_t patterns_per_thread = (patterns.size() + num_threads - 1) / num_threads;
    
    std::vector<std::future<std::vector<BatchResult>>> futures;
    futures.reserve(num_threads);
    
    for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        const size_t start_idx = thread_idx * patterns_per_thread;
        const size_t end_idx = std::min(start_idx + patterns_per_thread, patterns.size());
        
        if (start_idx >= patterns.size()) break;
        
        auto future = std::async(std::launch::async, [this, &patterns, start_idx, end_idx]() {
            std::vector<BatchResult> thread_results;
            thread_results.reserve(end_idx - start_idx);
            
            for (size_t i = start_idx; i < end_idx && !cancel_requested_; ++i) {
                auto start_time = std::chrono::high_resolution_clock::now();
                BatchResult result = process_single_pattern(patterns[i]);
                auto end_time = std::chrono::high_resolution_clock::now();
                
                double processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                update_stats(result.success, processing_time_ms);
                
                thread_results.push_back(std::move(result));
                
                if (config_.fail_fast && !result.success) {
                    cancel_requested_ = true;
                    break;
                }
            }
            
            return thread_results;
        });
        
        futures.push_back(std::move(future));
    }
    
    // Collect results from all threads
    for (auto& future : futures) {
        auto thread_results = future.get();
        results.insert(results.end(), thread_results.begin(), thread_results.end());
    }
    
    return results;
}

std::future<std::vector<BatchResult>> BatchProcessor::process_batch_async(
    const std::vector<BatchPattern>& patterns,
    std::function<void(const BatchResult&)> callback) {
    
    return std::async(std::launch::async, [this, patterns, callback]() {
        auto results = process_batch(patterns);
        
        if (callback) {
            for (const auto& result : results) {
                callback(result);
            }
        }
        
        return results;
    });
}

void BatchProcessor::cancel_batch() {
    cancel_requested_ = true;
}

BatchProcessor::BatchStats BatchProcessor::get_batch_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void BatchProcessor::reset_batch_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = BatchStats{};
}

void BatchProcessor::update_config(const BatchConfig& config) {
    config_ = config;
}

BatchConfig BatchProcessor::get_config() const {
    return config_;
}

BatchResult BatchProcessor::process_single_pattern(const BatchPattern& pattern) {
    try {
        // Create a temporary DSL program with the pattern
        std::ostringstream program;
        program << "pattern " << pattern.pattern_id << " {\n";
        
        // Add input variable bindings
        for (const auto& [name, value] : pattern.inputs) {
            program << "    " << name << " = " << value << "\n";
        }
        
        // Add the pattern code
        program << "    " << pattern.pattern_code << "\n";
        program << "}\n";
        
        // Execute the pattern using the engine facade
        // For now, simulate execution with basic coherence analysis
        // In a real implementation, this would use the DSL interpreter
        
        auto& engine = sep::engine::EngineFacade::getInstance();
        
        // Extract a simple metric from the pattern code
        // This is a simplified implementation - real version would parse and execute DSL
        double coherence_value = 0.0;
        
        if (pattern.pattern_code.find("measure_coherence") != std::string::npos) {
            // Simulate coherence measurement - use simplified version for batch
            coherence_value = 0.6 + (rand() % 40) / 100.0;  // Random value 0.6-1.0
        } else if (pattern.pattern_code.find("measure_entropy") != std::string::npos) {
            // Simulate entropy measurement  
            coherence_value = 0.3 + (rand() % 70) / 100.0;  // Random value 0.3-1.0
        } else {
            // Default: use QFH analysis simulation
            coherence_value = 0.5 + (rand() % 50) / 100.0;  // Random value 0.5-1.0
        }
        
        return BatchResult(pattern.pattern_id, true, coherence_value);
        
    } catch (const std::exception& e) {
        return BatchResult(pattern.pattern_id, false, 0.0, e.what());
    } catch (...) {
        return BatchResult(pattern.pattern_id, false, 0.0, "Unknown error during pattern processing");
    }
}

void BatchProcessor::update_stats(bool success, double processing_time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.patterns_processed++;
    if (success) {
        stats_.patterns_succeeded++;
    } else {
        stats_.patterns_failed++;
    }
    
    stats_.total_processing_time_ms += processing_time_ms;
    stats_.average_processing_time_ms = stats_.total_processing_time_ms / stats_.patterns_processed;
}

} // namespace sep::engine::batch
