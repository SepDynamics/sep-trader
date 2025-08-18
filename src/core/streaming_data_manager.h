#pragma once

#include "core/pattern.h"
#include "candle_data.h"
#include "util/result.h"
#include <vector>
#include <string>
#include "core/standard_includes.h"
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

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
    using PatternCallback = std::function<void(const core::Pattern&)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    StreamingDataManager();
    ~StreamingDataManager();

    // Lifecycle management
    core::Result initialize();
    core::Result shutdown();

    // Stream management
    core::Result createStream(const StreamConfiguration& config);
    core::Result startStream(const std::string& stream_id);
    core::Result stopStream(const std::string& stream_id);
    core::Result deleteStream(const std::string& stream_id);

    // Data ingestion
    core::Result ingestData(const std::string& stream_id, const StreamDataPoint& data);
    core::Result ingestBatch(const std::string& stream_id, const std::vector<StreamDataPoint>& batch);

    // Real-time callbacks
    void setDataCallback(const std::string& stream_id, DataCallback callback);
    void setPatternCallback(const std::string& stream_id, PatternCallback callback);
    void setErrorCallback(const std::string& stream_id, ErrorCallback callback);

    // Stream queries
    std::vector<StreamDataPoint> getRecentData(const std::string& stream_id, 
                                              size_t count = 100);
    std::vector<core::Pattern> getRecentPatterns(const std::string& stream_id, 
                                                size_t count = 10);
    StreamStats getStreamStats(const std::string& stream_id);
    std::vector<std::string> getActiveStreams() const;

    // Pattern analysis integration
    core::Result analyzeStreamPattern(const std::string& stream_id, 
                                     const std::vector<StreamDataPoint>& window,
                                     core::Pattern& result);

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
        std::vector<core::Pattern> pattern_buffer;
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
