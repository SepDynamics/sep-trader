#pragma once

#include <string>
#include <vector>
#include <memory>
#include <hiredis/hiredis.h>
#include "core/timeframe.h"
#include <future>
#include <chrono>
#include "util/nlohmann_json_safe.h"

namespace sep {
namespace trading {

// Forward declarations for data types
struct MarketData {
    std::string symbol;
    double price;
    double volume;
    std::string timestamp;
};

struct TrainingData {
    std::string pair;
    std::vector<double> features;
    std::vector<double> targets;
    std::string timestamp;
    std::string metadata;
    std::chrono::system_clock::time_point timestamp_point;
    double target;
};

struct ModelState {
    std::string model_id;
    std::string pair;
    std::vector<uint8_t> weights;
    double accuracy;
    std::chrono::system_clock::time_point trained_at;
    nlohmann::json hyperparameters;
};

using ::sep::Timeframe;

class RemoteDataManager {
public:
    RemoteDataManager();
    ~RemoteDataManager();

    // Market data operations using canonical schema
    bool save_market_data(const MarketData& data);
    std::vector<MarketData> get_market_data(const std::string& pair, const Timeframe& timeframe, int limit);

    // Training data management using canonical schema
    bool save_training_data(const TrainingData& data);
    std::vector<TrainingData> get_training_data(const std::string& pair, int limit);

    // Model cache operations using canonical schema
    bool cache_model(const std::string& model_id, const std::string& model_data);
    std::string get_cached_model(const std::string& model_id);
    
    // Async operations
    std::future<std::vector<TrainingData>> fetch_training_data_async(
        const std::string& pair,
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end);
        
    std::future<bool> upload_model_async(const ModelState& model);
    
    // Connection management
    bool test_connection();
    bool connect_to_valkey();
    void disconnect_from_valkey();

private:
    redisContext* valkey_conn_;
    std::string valkey_url_;
    
    std::string get_market_data_key(const std::string& pair) const;
    std::string get_training_data_key(const std::string& pair) const;
    std::string get_model_cache_key(const std::string& model_id) const;
};

}
}
