#include "streaming_data_manager.h"
#include "core/facade.h"
#include "core/qfh.h"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace sep::engine::streaming {

StreamingDataManager::StreamingDataManager() = default;

StreamingDataManager::~StreamingDataManager() {
    shutdown();
}

core::Result<void> StreamingDataManager::initialize() {
    if (initialized_.load()) {
        return core::Result<void>(sep::Error(sep::Error::Code::AlreadyExists, "Already initialized"));
    }
    
    std::cout << "StreamingDataManager: Initializing real-time data processing" << std::endl;
    initialized_.store(true);
    shutdown_requested_.store(false);
    
    return core::Result<void>();
}

core::Result<void> StreamingDataManager::shutdown() {
    if (!initialized_.load()) {
        return core::Result<void>(sep::Error(sep::Error::Code::NotInitialized, "Not initialized"));
    }
    
    std::cout << "StreamingDataManager: Shutting down streams..." << std::endl;
    shutdown_requested_.store(true);
    
    // Stop all active streams
    std::lock_guard<std::mutex> lock(streams_mutex_);
    for (auto& [stream_id, context] : streams_) {
        if (context->active.load()) {
            context->active.store(false);
            context->data_available.notify_all();
            
            if (context->worker_thread.joinable()) {
                context->worker_thread.join();
            }
        }
    }
    
    streams_.clear();
    initialized_.store(false);
    
    return core::Result<void>();
}

core::Result<void> StreamingDataManager::createStream(const StreamConfiguration& config) {
    if (!initialized_.load()) {
        return core::Result<void>(sep::Error(sep::Error::Code::NotInitialized, "Not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    if (streams_.find(config.stream_id) != streams_.end()) {
        return core::Result<void>(sep::Error(sep::Error::Code::OperationFailed, "ALREADY_EXISTS"));
    }
    
    auto context = std::make_unique<StreamContext>();
    context->config = config;
    context->stats.last_update = std::chrono::system_clock::now();
    
    std::cout << "StreamingDataManager: Created stream '" << config.stream_id 
              << "' type=" << config.source_type << std::endl;
    
    streams_[config.stream_id] = std::move(context);
    
    return core::Result<void>();
}

core::Result<void> StreamingDataManager::startStream(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return core::Result<void>(sep::Error(sep::Error::Code::OperationFailed, "NOT_FOUND"));
    }
    
    auto& context = it->second;
    if (context->active.load()) {
        return core::Result<void>(sep::Error(sep::Error::Code::OperationFailed, "ALREADY_EXISTS"));
    }
    
    context->active.store(true);
    context->worker_thread = std::thread(&StreamingDataManager::processStreamData, this, context.get());
    
    std::cout << "StreamingDataManager: Started stream '" << stream_id << "'" << std::endl;
    
    return core::Result<void>();
}

core::Result<void> StreamingDataManager::stopStream(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return core::Result<void>(sep::Error(sep::Error::Code::OperationFailed, "NOT_FOUND"));
    }
    
    auto& context = it->second;
    if (!context->active.load()) {
        return core::Result<void>(sep::Error(sep::Error::Code::OperationFailed, "NOT_FOUND"));
    }
    
    context->active.store(false);
    context->data_available.notify_all();
    
    if (context->worker_thread.joinable()) {
        context->worker_thread.join();
    }
    
    std::cout << "StreamingDataManager: Stopped stream '" << stream_id << "'" << std::endl;
    
    return core::Result<void>();
}

core::Result<void> StreamingDataManager::deleteStream(const std::string& stream_id) {
    // First stop the stream if it's running
    stopStream(stream_id);
    
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return core::Result<void>(sep::Error(sep::Error::Code::OperationFailed, "NOT_FOUND"));
    }
    
    streams_.erase(it);
    std::cout << "StreamingDataManager: Deleted stream '" << stream_id << "'" << std::endl;
    
    return core::Result<void>();
}

core::Result<void> StreamingDataManager::ingestData(const std::string& stream_id,
                                            const StreamDataPoint& data) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return core::Result<void>(sep::Error(sep::Error::Code::OperationFailed, "NOT_FOUND"));
    }
    
    auto& context = it->second;
    
    {
        std::lock_guard<std::mutex> buffer_lock(context->buffer_mutex);
        
        // Check buffer overflow
        if (context->data_buffer.size() >= context->config.buffer_size) {
            context->data_buffer.pop(); // Remove oldest
            context->stats.buffer_overflows++;
        }
        
        context->data_buffer.push(data);
        context->stats.total_data_points++;
        context->stats.last_update = std::chrono::system_clock::now();
    }
    
    // Notify worker thread
    context->data_available.notify_one();
    
    // Call data callback if set
    if (context->data_callback) {
        context->data_callback(data);
    }
    
    return core::Result<void>();
}

core::Result<void> StreamingDataManager::ingestBatch(const std::string& stream_id,
                                             const std::vector<StreamDataPoint>& batch) {
    for (const auto& data_point : batch) {
        auto result = ingestData(stream_id, data_point);
        if (!result.isSuccess()) {
            return result;
        }
    }
    
    return core::Result<void>();
}

void StreamingDataManager::setDataCallback(const std::string& stream_id, DataCallback callback) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        it->second->data_callback = std::move(callback);
    }
}

void StreamingDataManager::setPatternCallback(const std::string& stream_id, PatternCallback callback) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        it->second->pattern_callback = std::move(callback);
    }
}

void StreamingDataManager::setErrorCallback(const std::string& stream_id, ErrorCallback callback) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        it->second->error_callback = std::move(callback);
    }
}

std::vector<StreamDataPoint> StreamingDataManager::getRecentData(const std::string& stream_id, 
                                                                size_t count) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return {};
    }
    
    auto& context = it->second;
    std::lock_guard<std::mutex> buffer_lock(context->buffer_mutex);
    
    std::vector<StreamDataPoint> result;
    result.reserve(std::min(count, context->data_buffer.size()));
    
    // Convert queue to vector (most recent last)
    auto temp_queue = context->data_buffer;
    while (!temp_queue.empty() && result.size() < count) {
        result.push_back(temp_queue.front());
        temp_queue.pop();
    }
    
    return result;
}

std::vector<core::Pattern> StreamingDataManager::getRecentPatterns(const std::string& stream_id, 
                                                                  size_t count) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return {};
    }
    
    auto& context = it->second;
    std::lock_guard<std::mutex> buffer_lock(context->buffer_mutex);
    
    size_t start_index = context->pattern_buffer.size() > count ? 
                        context->pattern_buffer.size() - count : 0;
    
    return std::vector<core::Pattern>(
        context->pattern_buffer.begin() + start_index,
        context->pattern_buffer.end()
    );
}

StreamStats StreamingDataManager::getStreamStats(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return {};
    }
    
    std::lock_guard<std::mutex> buffer_lock(it->second->buffer_mutex);
    return it->second->stats;
}

std::vector<std::string> StreamingDataManager::getActiveStreams() const {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    std::vector<std::string> active_streams;
    for (const auto& [stream_id, context] : streams_) {
        if (context->active.load()) {
            active_streams.push_back(stream_id);
        }
    }
    
    return active_streams;
}

core::Result<void> StreamingDataManager::analyzeStreamPattern(const std::string& stream_id,
                                                       const std::vector<StreamDataPoint>& window,
                                                       core::Pattern& result) {
    if (window.empty()) {
        return core::Result<void>(sep::Error(sep::Error::Code::OperationFailed, "INVALID_ARGUMENT"));
    }
    
    // Combine all data streams into one bitstream for analysis
    std::vector<uint8_t> combined_bitstream;
    for (const auto& data_point : window) {
        combined_bitstream.insert(combined_bitstream.end(), 
                                data_point.data_stream.begin(), 
                                data_point.data_stream.end());
    }
    
    // Use QFH analysis through engine facade
    auto& engine = sep::engine::EngineFacade::getInstance();
    
    sep::engine::QFHAnalysisRequest qfh_request;
    qfh_request.bitstream = combined_bitstream;
    
    sep::engine::QFHAnalysisResponse qfh_response;
    auto qfh_result = engine.qfhAnalyze(qfh_request, qfh_response);
    
    if (!qfh_result.isSuccess()) {
        return qfh_result;
    }
    
    // Build pattern from QFH analysis
    result.id = static_cast<uint32_t>(std::hash<std::string>{}(stream_id + "_pattern_" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    )));
    result.coherence = qfh_response.coherence;
    result.quantum_state.coherence = qfh_response.coherence;
    result.quantum_state.stability = 1.0f - qfh_response.rupture_ratio;
    result.quantum_state.entropy = qfh_response.entropy;
    
    return core::Result<void>();
}

void StreamingDataManager::processStreamData(StreamContext* context) {
    std::cout << "StreamingDataManager: Worker thread started for stream '" 
              << context->config.stream_id << "'" << std::endl;
    
    std::vector<StreamDataPoint> analysis_window;
    const size_t window_size = 10; // Analyze every 10 data points
    
    while (context->active.load() && !shutdown_requested_.load()) {
        std::unique_lock<std::mutex> lock(context->buffer_mutex);
        
        // Wait for data or shutdown
        context->data_available.wait(lock, [context, this]() {
            return !context->data_buffer.empty() || 
                   !context->active.load() || 
                   shutdown_requested_.load();
        });
        
        if (!context->active.load() || shutdown_requested_.load()) {
            break;
        }
        
        // Process available data
        while (!context->data_buffer.empty() && analysis_window.size() < window_size) {
            analysis_window.push_back(context->data_buffer.front());
            context->data_buffer.pop();
        }
        
        lock.unlock();
        
        // Analyze window if we have enough data
        if (analysis_window.size() >= window_size) {
            analyzeDataWindow(context, analysis_window);
            analysis_window.clear();
        }
        
        // Sleep to respect sample rate
        std::this_thread::sleep_for(context->config.sample_rate);
    }
    
    std::cout << "StreamingDataManager: Worker thread stopped for stream '" 
              << context->config.stream_id << "'" << std::endl;
}

void StreamingDataManager::analyzeDataWindow(StreamContext* context, 
                                           const std::vector<StreamDataPoint>& window) {
    if (!context->config.real_time_analysis) {
        return;
    }
    
    try {
        core::Pattern pattern;
        auto result = analyzeStreamPattern(context->config.stream_id, window, pattern);
        
        if (result.isSuccess()) {
            // Update stream statistics
            {
                std::lock_guard<std::mutex> lock(context->buffer_mutex);
                context->stats.processed_patterns++;
                context->stats.average_coherence = 
                    (context->stats.average_coherence * (context->stats.processed_patterns - 1) + 
                     pattern.coherence) / context->stats.processed_patterns;
                context->stats.average_entropy = 
                    (context->stats.average_entropy * (context->stats.processed_patterns - 1) + 
                     pattern.quantum_state.entropy) / context->stats.processed_patterns;
                
                // Store pattern in buffer
                context->pattern_buffer.push_back(pattern);
                
                // Limit pattern buffer size
                if (context->pattern_buffer.size() > 100) {
                    context->pattern_buffer.erase(context->pattern_buffer.begin());
                }
            }
            
            // Call pattern callback if set
            if (context->pattern_callback) {
                context->pattern_callback(pattern);
            }
            
            std::cout << "StreamingDataManager: Analyzed pattern for stream '" 
                      << context->config.stream_id << "' coherence=" 
                      << pattern.coherence << std::endl;
        }
    } catch (const std::exception& e) {
        std::string error_msg = "Pattern analysis failed: " + std::string(e.what());
        std::cout << "StreamingDataManager: " << error_msg << std::endl;
        
        if (context->error_callback) {
            context->error_callback(error_msg);
        }
    }
}

void StreamingDataManager::setBufferSize(const std::string& stream_id, size_t size) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        it->second->config.buffer_size = size;
    }
}

void StreamingDataManager::clearBuffer(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> buffer_lock(it->second->buffer_mutex);
        
        // Clear data buffer
        std::queue<StreamDataPoint> empty_queue;
        it->second->data_buffer.swap(empty_queue);
        
        // Clear pattern buffer
        it->second->pattern_buffer.clear();
        
        // Reset stats
        it->second->stats.total_data_points = 0;
        it->second->stats.processed_patterns = 0;
        it->second->stats.buffer_overflows = 0;
        it->second->stats.average_coherence = 0.0f;
        it->second->stats.average_entropy = 0.0f;
    }
}

float StreamingDataManager::getBufferUtilization(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return 0.0f;
    }
    
    std::lock_guard<std::mutex> buffer_lock(it->second->buffer_mutex);
    return static_cast<float>(it->second->data_buffer.size()) / 
           static_cast<float>(it->second->config.buffer_size);
}

} // namespace sep::engine::streaming
