#pragma once

#include "core/standard_includes.h"
#include "core/result_types.h"
#include "core/types.h"

#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include "candle_data.h"
#include "core/pattern.h"
#include "core/quantum_types.h"

namespace sep::engine::streaming {

/// Real-time streaming data types for DSL integration
struct StreamDataPoint {
    std::chrono::system_clock::time_point timestamp;
    std::string source_id;
    std::vector<uint8_t> data_stream;
    float coherence{0.0f};
    float entropy{0.0f};
    std::string metadata;
};

struct StreamConfiguration {
    std::string stream_id;
    std::string source_type;  // "oanda", "market_data", "sensor", "file"
    std::string endpoint;
    std::vector<std::string> instruments;
    size_t buffer_size{1000};
    std::chrono::milliseconds sample_rate{100};
    bool real_time_analysis{true};
    float coherence_threshold{0.5f};
};

struct StreamStats {
    uint64_t total_data_points{0};
    uint64_t processed_patterns{0};
    uint64_t buffer_overflows{0};
    float average_coherence{0.0f};
    float average_entropy{0.0f};
    std::chrono::system_clock::time_point last_update;
};

/// Real-time streaming data manager for DSL pattern analysis
class StreamingDataManager {
public:
    using DataCallback = std::function<void(const StreamDataPoint&)>;
    using PatternCallback = std::function<void(const quantum::Pattern&)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    StreamingDataManager();
    ~StreamingDataManager();

    // Lifecycle management
    sep::Result<void> initialize();
    sep::Result<void> shutdown();

    // Stream management
    sep::Result<void> createStream(const StreamConfiguration& config);
    sep::Result<void> startStream(const std::string& stream_id);
    sep::Result<void> stopStream(const std::string& stream_id);
    sep::Result<void> deleteStream(const std::string& stream_id);

    // Data ingestion
    sep::Result<void> ingestData(const std::string& stream_id, const StreamDataPoint& data);
    sep::Result<void> ingestBatch(const std::string& stream_id,
                                  const std::vector<StreamDataPoint>& batch);

    // Real-time callbacks
    void setDataCallback(const std::string& stream_id, DataCallback callback);
    void setPatternCallback(const std::string& stream_id, PatternCallback callback);
    void setErrorCallback(const std::string& stream_id, ErrorCallback callback);

    // Stream queries
    std::vector<StreamDataPoint> getRecentData(const std::string& stream_id, 
                                              size_t count = 100);
    std::vector<quantum::Pattern> getRecentPatterns(const std::string& stream_id,
                                                    size_t count = 10);
    StreamStats getStreamStats(const std::string& stream_id);
    std::vector<std::string> getActiveStreams() const;

    // Pattern analysis integration
    sep::Result<void> analyzeStreamPattern(const std::string& stream_id,
                                           const std::vector<StreamDataPoint>& window,
                                           quantum::Pattern& result);

    // Buffer management
    void setBufferSize(const std::string& stream_id, size_t size);
    void clearBuffer(const std::string& stream_id);
    float getBufferUtilization(const std::string& stream_id);

    // Thread safety
    StreamingDataManager(const StreamingDataManager&) = delete;
    StreamingDataManager& operator=(const StreamingDataManager&) = delete;

private:
    struct StreamContext {
        StreamConfiguration config;
        std::queue<StreamDataPoint> data_buffer;
        std::vector<quantum::Pattern> pattern_buffer;
        StreamStats stats;
        std::atomic<bool> active{false};
        std::thread worker_thread;
        
        // Callbacks
        DataCallback data_callback;
        PatternCallback pattern_callback;
        ErrorCallback error_callback;
        
        // Synchronization
        mutable std::mutex buffer_mutex;
        std::condition_variable data_available;
    };

    // Stream storage
    std::unordered_map<std::string, std::unique_ptr<StreamContext>> streams_;
    mutable std::mutex streams_mutex_;
    
    // Worker thread management
    void processStreamData(StreamContext* context);
    void analyzeDataWindow(StreamContext* context, 
                          const std::vector<StreamDataPoint>& window);
    
    // Internal state
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
};

} // namespace sep::engine::streaming
