#pragma once

#include "pattern.h"
#include "candle_data.h"
#include "result.h"
#include "memory_tier_manager.hpp"
#include "types.h"
#include <vector>
#include <string>
#include <memory>

// The Engine Facade - The Brain That the API Layer Observes
// This is the clean measurement boundary between HTTP and core systems

namespace sep::engine {

// Forward declarations to hide implementation details from API
namespace memory {
    class MemoryTierManager;
}

/// Request types for the facade (pure data, no logic)
struct PatternProcessRequest {
    std::vector<::sep::quantum::Pattern> patterns;
    std::string context_id;
    float coherence_threshold{0.5f};
    bool async_processing{false};
};

struct PatternAnalysisRequest {
    std::string pattern_id;
    int analysis_depth{3};
    bool include_relationships{true};
};

struct MemoryQueryRequest {
    std::string query_id;
    int max_results{100};
    float relevance_threshold{0.3f};
};

struct QFHAnalysisRequest {
    std::vector<uint8_t> bitstream;
};

struct QFHAnalysisResponse {
    float rupture_ratio{0.0f};
    float flip_ratio{0.0f};
    bool collapse_detected{false};
    float coherence{0.0f};
    float entropy{0.0f};
};

struct ManifoldOptimizationRequest {
    std::string pattern_id;
    float target_coherence{0.8f};
    float target_stability{0.7f};
};

struct ManifoldOptimizationResponse {
    bool success{false};
    std::string pattern_id;
    float optimized_coherence{0.0f};
    float optimized_stability{0.0f};
};

struct StorePatternRequest {
    ::sep::quantum::Pattern pattern;
    float coherence{0.0f};
    float stability{0.0f};
    uint32_t generation_count{0};
    float weight{1.0f};
};

struct StorePatternResponse {
    bool success{false};
    std::string error_message;
};

struct BatchProcessRequest {
    std::vector<::sep::CandleData> market_data;
    std::string symbol;
    uint64_t timestamp_start{0};
    uint64_t timestamp_end{0};
};

// New advanced batch processing structures
struct AdvancedBatchRequest {
    std::vector<std::string> pattern_codes;
    std::vector<std::string> pattern_ids;
    std::vector<std::vector<std::pair<std::string, double>>> pattern_inputs;
    size_t max_parallel_threads{0};  // 0 = auto-detect
    size_t batch_size{100};
    bool fail_fast{false};
    double timeout_seconds{30.0};
};

struct BatchPatternResult {
    std::string pattern_id;
    bool success;
    double value;
    std::string error_message;
    double processing_time_ms;
};

struct AdvancedBatchResponse {
    std::vector<::sep::engine::BatchPatternResult> results;
    size_t patterns_processed{0};
    size_t patterns_succeeded{0};
    size_t patterns_failed{0};
    double total_processing_time_ms{0.0};
    double average_processing_time_ms{0.0};
};

// Engine configuration structures
struct ConfigSetRequest {
    std::string parameter_name;
    std::string value_type;  // "bool", "int", "double", "string"
    std::string value_string;
};

struct ConfigGetRequest {
    std::string parameter_name;
};

struct ConfigResponse {
    bool success{false};
    std::string parameter_name;
    std::string value_type;
    std::string value_string;
    std::string error_message;
};

struct ConfigListResponse {
    bool success{false};
    std::vector<std::string> parameter_names;
    std::vector<std::string> parameter_descriptions;
    std::vector<std::string> parameter_categories;
    std::vector<bool> requires_restart;
    std::string error_message;
};

// Streaming data support structures
struct StreamCreateRequest {
    std::string stream_id;
    std::string source_type;  // "oanda", "market_data", "sensor", "file"
    std::string endpoint;
    std::vector<std::string> instruments;
    size_t buffer_size{1000};
    uint32_t sample_rate_ms{100};
    bool real_time_analysis{true};
    float coherence_threshold{0.5f};
};

struct StreamDataRequest {
    std::string stream_id;
    std::vector<uint8_t> data_stream;
    std::string metadata;
};

struct StreamQueryRequest {
    std::string stream_id;
    size_t count{100};
    bool include_patterns{false};
};

/// Response types for the facade (pure data, no logic)
struct PatternProcessResponse {
    std::vector<::sep::quantum::Pattern> processed_patterns;
    std::string correlation_id;
    float coherence_score{0.0f};
    bool processing_complete{false};
};

struct PatternAnalysisResponse {
    ::sep::quantum::Pattern pattern;
    std::vector<::sep::quantum::PatternRelationship> relationships;
    float confidence_score{0.0f};
    float entropy{0.0f};
    std::string analysis_summary;
};

struct BitExtractionRequest {
    std::string pattern_id;
};

struct BitExtractionResponse {
    std::vector<uint8_t> bitstream;
    bool success{false};
    std::string error_message;
};

struct MemoryMetricsResponse {
    float stm_utilization{0.0f};
    float mtm_utilization{0.0f};
    float ltm_utilization{0.0f};
    uint64_t total_patterns{0};
    uint64_t active_patterns{0};
    float coherence_level{0.0f};
    
    // Pattern cache metrics
    size_t cached_patterns{0};
    size_t cache_hits{0};
    size_t cache_misses{0};
    float cache_hit_ratio{0.0f};
    
    // GPU memory pool metrics
    size_t gpu_total_allocated{0};
    size_t gpu_current_usage{0};
    size_t gpu_peak_usage{0};
    size_t gpu_fragmentation_ratio{0};
    size_t gpu_allocations{0};
    size_t gpu_deallocations{0};
};

// GPU memory management requests/responses
struct GPUMemoryAllocRequest {
    size_t size_bytes;
    size_t alignment{256};
    bool use_stream{false};
    uint64_t stream_id{0};
};

struct GPUMemoryAllocResponse {
    bool success{false};
    uint64_t memory_handle{0};  // Opaque handle for DSL
    size_t allocated_size{0};
    std::string error_message;
};

struct GPUMemoryDeallocRequest {
    uint64_t memory_handle;
    bool use_stream{false};
    uint64_t stream_id{0};
};

struct GPUMemoryConfigRequest {
    bool auto_defragment{true};
    float defragment_threshold{0.5f};
    bool auto_grow{true};
    size_t growth_factor{2};
    size_t initial_pool_size{256 * 1024 * 1024}; // 256MB
};

struct HealthStatusResponse {
    bool engine_healthy{false};
    bool quantum_systems_ready{false};
    bool memory_systems_ready{false};
    float cpu_usage{0.0f};
    float memory_usage{0.0f};
    std::string status_message;
};

struct StreamResponse {
    bool success{false};
    std::string stream_id;
    std::string error_message;
    std::vector<std::string> active_streams;
};

struct StreamDataResponse {
    std::vector<uint8_t> recent_data;
    std::vector<::sep::quantum::Pattern> recent_patterns;
    uint64_t total_data_points{0};
    uint64_t processed_patterns{0};
    float average_coherence{0.0f};
    float buffer_utilization{0.0f};
};

/// The Engine Facade - Simple Functions That Hide All Complexity
/// The API layer becomes a dumb observer that sends requests and receives responses
class EngineFacade {
public:
    // Lifecycle management
    static EngineFacade& getInstance();
    ::sep::core::Result initialize();
    ::sep::core::Result shutdown();
    
    // Core pattern operations
    ::sep::core::Result processPatterns(const PatternProcessRequest& request, 
                                PatternProcessResponse& response);
    
    ::sep::core::Result analyzePattern(const PatternAnalysisRequest& request,
                               PatternAnalysisResponse& response);

    ::sep::core::Result qfhAnalyze(const QFHAnalysisRequest& request,
                            QFHAnalysisResponse& response);

    ::sep::core::Result manifoldOptimize(const ManifoldOptimizationRequest& request,
    ManifoldOptimizationResponse& response);

    ::sep::core::Result extractBits(const BitExtractionRequest& request,
    BitExtractionResponse& response);

    ::sep::core::Result storePattern(const StorePatternRequest& request,
                            StorePatternResponse& response);
    
    ::sep::core::Result processBatch(const BatchProcessRequest& request,
                             PatternProcessResponse& response);
    
    // Advanced batch processing
    ::sep::core::Result processAdvancedBatch(const AdvancedBatchRequest& request,
                                     AdvancedBatchResponse& response);
    
    // Streaming data operations
    ::sep::core::Result createStream(const StreamCreateRequest& request,
                             StreamResponse& response);
    ::sep::core::Result startStream(const std::string& stream_id,
                            StreamResponse& response);
    ::sep::core::Result stopStream(const std::string& stream_id,
                           StreamResponse& response);
    ::sep::core::Result deleteStream(const std::string& stream_id,
                             StreamResponse& response);
    ::sep::core::Result ingestStreamData(const StreamDataRequest& request,
                                 StreamResponse& response);
    ::sep::core::Result queryStream(const StreamQueryRequest& request,
                            StreamDataResponse& response);
    
    // Memory operations  
    ::sep::core::Result queryMemory(const MemoryQueryRequest& request,
                            std::vector<::sep::quantum::Pattern>& results);
    
    // System status
    ::sep::core::Result getHealthStatus(HealthStatusResponse& response);
    ::sep::core::Result getMemoryMetrics(MemoryMetricsResponse& response);
    
    // Pattern cache operations
    ::sep::core::Result clearPatternCache();
    ::sep::core::Result configurePatternCache(size_t max_size, int ttl_minutes, float coherence_threshold);
    
    // GPU memory operations
    ::sep::core::Result allocateGPUMemory(const GPUMemoryAllocRequest& request,
                                  GPUMemoryAllocResponse& response);
    ::sep::core::Result deallocateGPUMemory(const GPUMemoryDeallocRequest& request);
    ::sep::core::Result configureGPUMemory(const GPUMemoryConfigRequest& request);
    ::sep::core::Result defragmentGPUMemory();
    ::sep::core::Result resetGPUMemoryStats();
    
    // Engine configuration operations
    ::sep::core::Result setEngineConfig(const ConfigSetRequest& request,
                                ConfigResponse& response);
    ::sep::core::Result getEngineConfig(const ConfigGetRequest& request,
                                ConfigResponse& response);
    ::sep::core::Result listEngineConfig(ConfigListResponse& response);
    ::sep::core::Result resetEngineConfig(const std::string& category = "");  // Empty = reset all
    
    // Prevent copying/moving
    EngineFacade(const EngineFacade&) = delete;
    EngineFacade& operator=(const EngineFacade&) = delete;
    EngineFacade(EngineFacade&&) = delete;
    EngineFacade& operator=(EngineFacade&&) = delete;

private:
    EngineFacade() = default;
    ~EngineFacade() = default;
    
    // Implementation details hidden from API layer
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool initialized_{false};
};

} // namespace sep::engine