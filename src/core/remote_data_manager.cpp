#include "RemoteDataManager.h"

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <thread>
#include <algorithm>
#include "core/sep_precompiled.h"

namespace sep::trading {

namespace {

std::string to_timestamp_string(const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    return std::to_string(static_cast<int64_t>(time_t));
}

std::chrono::system_clock::time_point from_timestamp_string(const std::string& ts_str) {
    try {
        int64_t ts = std::stoll(ts_str);
        return std::chrono::system_clock::from_time_t(static_cast<time_t>(ts));
    } catch (...) {
        return std::chrono::system_clock::now();
    }
}

} // namespace

RemoteDataManager::RemoteDataManager()
    : valkey_conn_(nullptr) {
    // Get Valkey URL from environment
    valkey_url_ = std::getenv("VALKEY_URL") ? std::getenv("VALKEY_URL") : "redis://localhost:6379";
    connect_to_valkey();
}

RemoteDataManager::~RemoteDataManager() {
    disconnect_from_valkey();
}

bool RemoteDataManager::connect_to_valkey() {
    if (valkey_conn_) {
        disconnect_from_valkey();
    }
    
    // Parse Redis URL (simplified - assumes redis://host:port format)
    std::string host = "localhost";
    int port = 6379;
    
    if (valkey_url_.find("redis://") == 0) {
        std::string url_part = valkey_url_.substr(8); // Remove "redis://"
        size_t colon_pos = url_part.find(':');
        if (colon_pos != std::string::npos) {
            host = url_part.substr(0, colon_pos);
            try {
                port = std::stoi(url_part.substr(colon_pos + 1));
            } catch (...) {
                port = 6379; // fallback
            }
        } else {
            host = url_part;
        }
    }
    
    valkey_conn_ = redisConnect(host.c_str(), port);
    
    if (!valkey_conn_ || valkey_conn_->err) {
        std::cerr << "[REMOTE_DATA] ❌ Failed to connect to Valkey at " << host << ":" << port;
        if (valkey_conn_) {
            std::cerr << " - " << valkey_conn_->errstr;
            redisFree(valkey_conn_);
            valkey_conn_ = nullptr;
        }
        std::cerr << std::endl;
        return false;
    }
    
    std::cout << "[REMOTE_DATA] 🔗 Connected to Valkey at " << host << ":" << port << std::endl;
    return true;
}

void RemoteDataManager::disconnect_from_valkey() {
    if (valkey_conn_) {
        redisFree(valkey_conn_);
        valkey_conn_ = nullptr;
    }
}

bool RemoteDataManager::test_connection() {
    if (!valkey_conn_) return false;
    
    redisReply* reply = (redisReply*)redisCommand(valkey_conn_, "PING");
    bool success = (reply && reply->type == REDIS_REPLY_STATUS && strcmp(reply->str, "PONG") == 0);
    if (reply) freeReplyObject(reply);
    
    return success;
}

std::string RemoteDataManager::get_market_data_key(const std::string& pair) const {
    return "md:price:" + pair;
}

std::string RemoteDataManager::get_training_data_key(const std::string& pair) const {
    return "training:data:" + pair;
}

std::string RemoteDataManager::get_model_cache_key(const std::string& model_id) const {
    return "model:cache:" + model_id;
}

bool RemoteDataManager::save_market_data(const MarketData& data) {
    if (!valkey_conn_) return false;
    
    std::string key = get_market_data_key(data.symbol);
    
    // Store only the available price and volume fields
    nlohmann::json candle_data;
    candle_data["c"] = data.price;   // Close price
    candle_data["v"] = data.volume;  // Volume

    std::string json_str = candle_data.dump();
    
    // Extract timestamp from data.timestamp string or use current time
    int64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Add to Valkey ZSET using canonical schema
    redisReply* reply = (redisReply*)redisCommand(valkey_conn_, 
        "ZADD %s %lld %s", key.c_str(), timestamp_ms, json_str.c_str());
    
    bool success = (reply && reply->type == REDIS_REPLY_INTEGER && reply->integer >= 0);
    if (reply) freeReplyObject(reply);
    
    if (!success) {
        std::cerr << "[REMOTE_DATA] Failed to save market data for " << data.symbol << std::endl;
    }
    
    return success;
}

std::vector<MarketData> RemoteDataManager::get_market_data(const std::string& pair, const Timeframe& timeframe, int limit) {
    std::vector<MarketData> results;
    
    if (!valkey_conn_) return results;
    
    std::string key = get_market_data_key(pair);
    
    // Get recent data from ZSET using canonical schema
    redisReply* reply = (redisReply*)redisCommand(valkey_conn_, 
        "ZREVRANGE %s 0 %d WITHSCORES", key.c_str(), limit - 1);
    
    if (reply && reply->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < reply->elements; i += 2) {
            try {
                std::string json_data(reply->element[i]->str, reply->element[i]->len);
                std::string score_str(reply->element[i+1]->str, reply->element[i+1]->len);
                
                nlohmann::json ohlc_data = nlohmann::json::parse(json_data);
                
                MarketData market_data;
                market_data.symbol = pair;
                market_data.price = ohlc_data.value("c", 0.0); // Use close price
                market_data.volume = ohlc_data.value("v", 0.0);
                market_data.timestamp = score_str; // Unix timestamp as string
                
                results.push_back(market_data);
            } catch (const std::exception& e) {
                std::cerr << "[REMOTE_DATA] Error parsing market data: " << e.what() << std::endl;
            }
        }
    }
    
    if (reply) freeReplyObject(reply);
    return results;
}

bool RemoteDataManager::save_training_data(const TrainingData& data) {
    if (!valkey_conn_) return false;
    
    std::string key = get_training_data_key(data.pair);
    
    nlohmann::json training_json;
    training_json["features"] = data.features;
    training_json["targets"] = data.targets;
    training_json["metadata"] = data.metadata;
    
    std::string json_str = training_json.dump();
    
    int64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        data.timestamp_point.time_since_epoch()).count();
    
    // Store in ZSET with timestamp score
    redisReply* reply = (redisReply*)redisCommand(valkey_conn_,
        "ZADD %s %lld %s", key.c_str(), timestamp_ms, json_str.c_str());
    
    bool success = (reply && reply->type == REDIS_REPLY_INTEGER && reply->integer >= 0);
    if (reply) freeReplyObject(reply);
    
    return success;
}

std::vector<TrainingData> RemoteDataManager::get_training_data(const std::string& pair, int limit) {
    std::vector<TrainingData> results;
    
    if (!valkey_conn_) return results;
    
    std::string key = get_training_data_key(pair);
    
    redisReply* reply = (redisReply*)redisCommand(valkey_conn_,
        "ZREVRANGE %s 0 %d WITHSCORES", key.c_str(), limit - 1);
    
    if (reply && reply->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < reply->elements; i += 2) {
            try {
                std::string json_data(reply->element[i]->str, reply->element[i]->len);
                std::string score_str(reply->element[i+1]->str, reply->element[i+1]->len);
                
                nlohmann::json training_json = nlohmann::json::parse(json_data);
                
                TrainingData training_data;
                training_data.pair = pair;
                training_data.timestamp = score_str;
                training_data.timestamp_point = from_timestamp_string(score_str);
                training_data.metadata = training_json.value("metadata", "");
                
                if (training_json.contains("features") && training_json["features"].is_array()) {
                    for (const auto& feature : training_json["features"]) {
                        training_data.features.push_back(feature.get<double>());
                    }
                }
                
                if (training_json.contains("targets") && training_json["targets"].is_array()) {
                    for (const auto& target : training_json["targets"]) {
                        training_data.targets.push_back(target.get<double>());
                    }
                }
                
                // Set single target for compatibility
                if (!training_data.targets.empty()) {
                    training_data.target = training_data.targets[0];
                }
                
                results.push_back(training_data);
            } catch (const std::exception& e) {
                std::cerr << "[REMOTE_DATA] Error parsing training data: " << e.what() << std::endl;
            }
        }
    }
    
    if (reply) freeReplyObject(reply);
    return results;
}

bool RemoteDataManager::cache_model(const std::string& model_id, const std::string& model_data) {
    if (!valkey_conn_) return false;
    
    std::string key = get_model_cache_key(model_id);
    
    // Store model data with 24 hour expiration
    redisReply* reply = (redisReply*)redisCommand(valkey_conn_,
        "SETEX %s %d %s", key.c_str(), 86400, model_data.c_str());
    
    bool success = (reply && reply->type == REDIS_REPLY_STATUS && strcmp(reply->str, "OK") == 0);
    if (reply) freeReplyObject(reply);
    
    return success;
}

std::string RemoteDataManager::get_cached_model(const std::string& model_id) {
    if (!valkey_conn_) return "";
    
    std::string key = get_model_cache_key(model_id);
    
    redisReply* reply = (redisReply*)redisCommand(valkey_conn_, "GET %s", key.c_str());
    
    std::string model_data;
    if (reply && reply->type == REDIS_REPLY_STRING) {
        model_data = std::string(reply->str, reply->len);
    }
    
    if (reply) freeReplyObject(reply);
    return model_data;
}

std::future<std::vector<TrainingData>> RemoteDataManager::fetch_training_data_async(
    const std::string& pair,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end) {
    
    return std::async(std::launch::async, [this, pair, start, end]() {
        std::vector<TrainingData> results;
        
        if (!valkey_conn_) return results;
        
        std::string key = get_training_data_key(pair);
        
        int64_t start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count();
        int64_t end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch()).count();
        
        // Get data in time range
        redisReply* reply = (redisReply*)redisCommand(valkey_conn_,
            "ZRANGEBYSCORE %s %lld %lld WITHSCORES", key.c_str(), start_ms, end_ms);
        
        if (reply && reply->type == REDIS_REPLY_ARRAY) {
            for (size_t i = 0; i < reply->elements; i += 2) {
                try {
                    std::string json_data(reply->element[i]->str, reply->element[i]->len);
                    std::string score_str(reply->element[i+1]->str, reply->element[i+1]->len);
                    
                    nlohmann::json training_json = nlohmann::json::parse(json_data);
                    
                    TrainingData training_data;
                    training_data.pair = pair;
                    training_data.timestamp = score_str;
                    training_data.timestamp_point = from_timestamp_string(score_str);
                    training_data.metadata = training_json.value("metadata", "");
                    
                    if (training_json.contains("features") && training_json["features"].is_array()) {
                        for (const auto& feature : training_json["features"]) {
                            training_data.features.push_back(feature.get<double>());
                        }
                    }
                    
                    if (training_json.contains("targets") && training_json["targets"].is_array()) {
                        for (const auto& target : training_json["targets"]) {
                            training_data.targets.push_back(target.get<double>());
                        }
                    }
                    
                    if (!training_data.targets.empty()) {
                        training_data.target = training_data.targets[0];
                    }
                    
                    results.push_back(training_data);
                } catch (const std::exception& e) {
                    std::cerr << "[REMOTE_DATA] Error parsing training data: " << e.what() << std::endl;
                }
            }
        }
        
        if (reply) freeReplyObject(reply);
        return results;
    });
}

std::future<bool> RemoteDataManager::upload_model_async(const ModelState& model) {
    return std::async(std::launch::async, [this, model]() {
        if (!valkey_conn_) return false;
        
        // Store model metadata
        std::string model_key = get_model_cache_key(model.model_id);
        
        nlohmann::json model_json;
        model_json["pair"] = model.pair;
        model_json["accuracy"] = model.accuracy;
        model_json["trained_at"] = to_timestamp_string(model.trained_at);
        model_json["hyperparameters"] = model.hyperparameters;
        
        std::string metadata_str = model_json.dump();
        
        // Store metadata with 7 day expiration
        redisReply* reply = (redisReply*)redisCommand(valkey_conn_,
            "SETEX %s %d %s", model_key.c_str(), 604800, metadata_str.c_str());
        
        bool metadata_success = (reply && reply->type == REDIS_REPLY_STATUS && strcmp(reply->str, "OK") == 0);
        if (reply) freeReplyObject(reply);
        
        // Store model weights separately
        std::string weights_key = model_key + ":weights";
        reply = (redisReply*)redisCommand(valkey_conn_,
            "SETEX %s %d %b", weights_key.c_str(), 604800, 
            model.weights.data(), model.weights.size());
        
        bool weights_success = (reply && reply->type == REDIS_REPLY_STATUS && strcmp(reply->str, "OK") == 0);
        if (reply) freeReplyObject(reply);
        
        return metadata_success && weights_success;
    });
}

} // namespace sep::trading