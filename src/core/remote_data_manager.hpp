#pragma once
#include "nlohmann_json_safe.h"

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "standard_includes.h"

namespace sep::trading {

struct DataSyncConfig {
    std::string remote_host = "localhost";
    int remote_port = 5432;
    std::string db_name = "sep_trading";
    std::string redis_host = "localhost";
    int redis_port = 6379;
    std::string data_path = "/opt/sep-data";
    
    // Local cache settings
    std::string local_cache_path = ".cache/remote_data";
    int cache_ttl_hours = 24;
    bool enable_compression = true;
};

struct TrainingData {
    std::string pair;
    std::chrono::system_clock::time_point timestamp;
    std::vector<double> features;
    double target;
    std::string metadata;
};

struct ModelState {
    std::string model_id;
    std::string pair;
    std::vector<uint8_t> weights;
    double accuracy;
    std::chrono::system_clock::time_point trained_at;
    nlohmann::json hyperparameters;
};

class RemoteDataManager {
public:
    explicit RemoteDataManager(const DataSyncConfig& config);
    ~RemoteDataManager();

    // Training data management
    std::future<std::vector<TrainingData>> fetch_training_data(
        const std::string& pair, 
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    );
    
    std::future<bool> upload_training_batch(const std::vector<TrainingData>& batch);
    
    // Model synchronization
    std::future<bool> upload_model(const ModelState& model);
    std::future<ModelState> download_latest_model(const std::string& pair);
    std::future<std::vector<ModelState>> list_available_models();
    
    // Real-time data streaming
    void start_streaming(const std::vector<std::string>& pairs);
    void stop_streaming();
    bool register_data_callback(std::function<void(const TrainingData&)> callback);
    
    // Cache management
    void clear_local_cache();
    size_t get_cache_size();
    bool is_cache_valid(const std::string& key);
    
    // Health and status
    bool test_connection();
    nlohmann::json get_remote_status();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Helper class for local/remote coordination
class TrainingCoordinator {
public:
    explicit TrainingCoordinator(std::shared_ptr<RemoteDataManager> remote_mgr);
    
    // Coordinate training between local and remote
    std::future<bool> start_distributed_training(
        const std::string& pair,
        const nlohmann::json& training_config
    );
    
    // Sync local training results to remote
    std::future<bool> sync_local_results(const std::string& session_id);
    
    // Download and apply remote model updates
    std::future<bool> pull_remote_updates();
    
    bool is_training_active() const;
    nlohmann::json get_training_status() const;
    
private:
    std::shared_ptr<RemoteDataManager> remote_manager_;
    std::string current_session_id_;
    std::atomic<bool> training_active_{false};
};

} // namespace sep::trading
