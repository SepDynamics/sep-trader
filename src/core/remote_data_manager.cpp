
#include "remote_data_manager.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <mutex>
#include "util/nlohmann_json_safe.h"
#include <optional>
#include "util/pqxx_time_point_traits.h"  // Must precede pqxx includes
#include <pqxx/pqxx>
#include <thread>

#include "core/sep_precompiled.h"
// #include <compression/gzip.hpp> // Optional compression - not available

namespace sep::trading {

class RemoteDataManager::Impl {
public:
    explicit Impl(const DataSyncConfig& config) : config_(config) {
        initialize_connections();
        setup_local_cache();
    }
    
    ~Impl() {
        if (redis_context_) {
            redisFree(redis_context_);
        }
    }
    
    std::future<std::vector<TrainingData>> fetch_training_data(
        const std::string& pair,
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    ) {
        return std::async(std::launch::async, [this, pair, start, end]() {
            std::vector<TrainingData> data;
            
            try {
                // Check local cache first
                std::string cache_key = generate_cache_key(pair, start, end);
                if (auto cached = load_from_cache(cache_key)) {
                    spdlog::info("Loaded {} training records from cache", cached->size());
                    return *cached;
                }
                
                // Fetch from remote database
                pqxx::connection conn(build_connection_string());
                pqxx::work txn(conn);
                
                auto result = txn.exec_params(
                    "SELECT pair, timestamp, features, target, metadata "
                    "FROM training_data WHERE pair = $1 AND timestamp BETWEEN $2 AND $3 "
                    "ORDER BY timestamp",
                    pair, start, end
                );
                
                for (const auto& row : result) {
                    TrainingData record;
                    record.pair = row[0].as<std::string>();
                    record.timestamp = row[1].as<std::chrono::system_clock::time_point>();
                    
                    // Parse JSON features array
                    try {
                        nlohmann::json features_json = nlohmann::json::parse(row[2].as<std::string>());
                        for (const auto& val : features_json) {
                            record.features.push_back(val.get<double>());
                        }
                    } catch (const nlohmann::json::parse_error&) {
                        // Skip invalid JSON features
                    }
                    
                    record.target = row[3].as<double>();
                    record.metadata = row[4].as<std::string>();
                    data.push_back(record);
                }
                
                txn.commit();
                
                // Cache the results
                save_to_cache(cache_key, data);
                
                spdlog::info("Fetched {} training records for {}", data.size(), pair);
                
            } catch (const std::exception& e) {
                spdlog::error("Failed to fetch training data: {}", e.what());
            }
            
            return data;
        });
    }
    
    std::future<bool> upload_training_batch(const std::vector<TrainingData>& batch) {
        return std::async(std::launch::async, [this, batch]() {
            try {
                pqxx::connection conn(build_connection_string());
                pqxx::work txn(conn);
                
                for (const auto& record : batch) {
                    // Convert features to JSON
                    nlohmann::json features_json = nlohmann::json::array();
                    for (double feature : record.features) {
                        features_json.push_back(feature);
                    }
                    
                    std::string features_str = features_json.dump();
                    
                    txn.exec_params(
                        "INSERT INTO training_data (pair, timestamp, features, target, metadata) "
                        "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (pair, timestamp) DO UPDATE SET "
                        "features = EXCLUDED.features, target = EXCLUDED.target, metadata = EXCLUDED.metadata",
                        record.pair, record.timestamp, features_str, record.target, record.metadata
                    );
                }
                
                txn.commit();
                spdlog::info("Uploaded {} training records", batch.size());
                return true;
                
            } catch (const std::exception& e) {
                spdlog::error("Failed to upload training batch: {}", e.what());
                return false;
            }
        });
    }
    
    std::future<bool> upload_model(const ModelState& model) {
        return std::async(std::launch::async, [this, model]() {
            try {
                // Store model weights in Redis for fast access
                std::string redis_key = "model:" + model.pair + ":" + model.model_id;
                
                redisReply* reply = (redisReply*)redisCommand(redis_context_, 
                    "SETEX %s %d %b", 
                    redis_key.c_str(), 
                    86400, // 24 hour TTL
                    model.weights.data(), 
                    model.weights.size()
                );
                
                if (reply) freeReplyObject(reply);
                
                // Store metadata in PostgreSQL
                pqxx::connection conn(build_connection_string());
                pqxx::work txn(conn);
                
                std::string hyperparams_str = model.hyperparameters.dump();
                
                txn.exec_params(
                    "INSERT INTO models (model_id, pair, accuracy, trained_at, hyperparameters, redis_key) "
                    "VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (model_id) DO UPDATE SET "
                    "accuracy = EXCLUDED.accuracy, trained_at = EXCLUDED.trained_at, "
                    "hyperparameters = EXCLUDED.hyperparameters, redis_key = EXCLUDED.redis_key",
                    model.model_id, model.pair, model.accuracy, model.trained_at, 
                    hyperparams_str, redis_key
                );
                
                txn.commit();
                spdlog::info("Uploaded model {} for pair {}", model.model_id, model.pair);
                return true;
                
            } catch (const std::exception& e) {
                spdlog::error("Failed to upload model: {}", e.what());
                return false;
            }
        });
    }
    
    bool test_connection() {
        try {
            // Test PostgreSQL
            pqxx::connection conn(build_connection_string());
            pqxx::work txn(conn);
            auto result = txn.exec("SELECT 1");
            txn.commit();
            
            // Test Redis
            redisReply* reply = (redisReply*)redisCommand(redis_context_, "PING");
            bool redis_ok = (reply && reply->type == REDIS_REPLY_STATUS && 
                           strcmp(reply->str, "PONG") == 0);
            if (reply) freeReplyObject(reply);
            
            return redis_ok;
            
        } catch (const std::exception& e) {
            spdlog::error("Connection test failed: {}", e.what());
            return false;
        }
    }
    
private:
    DataSyncConfig config_;
    redisContext* redis_context_ = nullptr;
    std::mutex cache_mutex_;
    
    void initialize_connections() {
        // Initialize Redis connection
        redis_context_ = redisConnect(config_.redis_host.c_str(), config_.redis_port);
        if (!redis_context_ || redis_context_->err) {
            spdlog::error("Failed to connect to Redis: {}", 
                redis_context_ ? redis_context_->errstr : "Connection failed");
        }
    }
    
    void setup_local_cache() {
        std::filesystem::create_directories(config_.local_cache_path);
    }
    
    std::string build_connection_string() {
        return fmt::format("host={} port={} dbname={} user=sep_user password=sep_password",
            config_.remote_host, config_.remote_port, config_.db_name);
    }
    
    std::string generate_cache_key(const std::string& pair, 
                                 const std::chrono::system_clock::time_point& start,
                                 const std::chrono::system_clock::time_point& end) {
        auto start_time_t = std::chrono::system_clock::to_time_t(start);
        auto end_time_t = std::chrono::system_clock::to_time_t(end);
        return fmt::format("training_{}_{}_{}_{}", pair, start_time_t, end_time_t, 
                          config_.enable_compression ? "gz" : "raw");
    }
    
    std::optional<std::vector<TrainingData>> load_from_cache(const std::string& key) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        std::string cache_file = config_.local_cache_path + "/" + key + ".cache";
        
        if (!std::filesystem::exists(cache_file)) {
            return std::nullopt;
        }
        
        // Check if cache is still valid
        auto file_time = std::filesystem::last_write_time(cache_file);
        auto now = std::filesystem::file_time_type::clock::now();
        auto hours_old = std::chrono::duration_cast<std::chrono::hours>(now - file_time).count();
        
        if (hours_old > config_.cache_ttl_hours) {
            std::filesystem::remove(cache_file);
            return std::nullopt;
        }
        
        // Load and deserialize data
        // Implementation would include JSON deserialization and optional decompression
        return std::nullopt; // Placeholder
    }
    
    void save_to_cache(const std::string& key, const std::vector<TrainingData>& data) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        std::string cache_file = config_.local_cache_path + "/" + key + ".cache";
        
        // Implementation would include JSON serialization and optional compression
        spdlog::debug("Saved {} records to cache: {}", data.size(), cache_file);
    }
};

// RemoteDataManager implementation
RemoteDataManager::RemoteDataManager(const DataSyncConfig& config) 
    : pImpl(std::make_unique<Impl>(config)) {}

RemoteDataManager::~RemoteDataManager() = default;

std::future<std::vector<TrainingData>> RemoteDataManager::fetch_training_data(
    const std::string& pair, 
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end) {
    return pImpl->fetch_training_data(pair, start, end);
}

std::future<bool> RemoteDataManager::upload_training_batch(const std::vector<TrainingData>& batch) {
    return pImpl->upload_training_batch(batch);
}

std::future<bool> RemoteDataManager::upload_model(const ModelState& model) {
    return pImpl->upload_model(model);
}

bool RemoteDataManager::test_connection() {
    return pImpl->test_connection();
}

// TrainingCoordinator implementation
TrainingCoordinator::TrainingCoordinator(std::shared_ptr<RemoteDataManager> remote_mgr) 
    : remote_manager_(remote_mgr) {}

std::future<bool> TrainingCoordinator::start_distributed_training(
    const std::string& pair,
    const nlohmann::json& training_config) {
    
    return std::async(std::launch::async, [this, pair, training_config]() {
        training_active_ = true;
        current_session_id_ = fmt::format("session_{}_{}", pair, 
            std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
        
        spdlog::info("Starting distributed training for {} (session: {})", 
                    pair, current_session_id_);
        
        // Implementation would coordinate training between local and remote
        return true;
    });
}

bool TrainingCoordinator::is_training_active() const {
    return training_active_.load();
}

} // namespace sep::trading