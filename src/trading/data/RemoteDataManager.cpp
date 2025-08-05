#include "trading/data/RemoteDataManager.h"
#include <iostream>
#include <stdexcept>

namespace sep {
namespace trading {

RemoteDataManager::RemoteDataManager(const std::string& postgres_conn_str, const std::string& redis_url) {
    try {
        pg_conn_ = std::make_unique<pqxx::connection>(postgres_conn_str);
        redis_conn_ = redisConnect("127.0.0.1", 6379);
        if (!redis_conn_ || redis_conn_->err) {
            throw std::runtime_error("Failed to connect to Redis");
        }
        std::cout << "Successfully connected to PostgreSQL and Redis." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error connecting to databases: " << e.what() << std::endl;
        throw;
    }
}

RemoteDataManager::~RemoteDataManager() {
    if (redis_conn_) {
        redisFree(redis_conn_);
    }
}

void RemoteDataManager::save_market_data(const MarketData& data) {
    try {
        pqxx::work txn(*pg_conn_);
        txn.exec_params(
            "INSERT INTO market_data (symbol, price, volume, timestamp) VALUES ($1, $2, $3, $4)",
            data.symbol, data.price, data.volume, data.timestamp
        );
        txn.commit();
    } catch (const std::exception& e) {
        std::cerr << "Error saving market data: " << e.what() << std::endl;
        throw;
    }
}

std::vector<MarketData> RemoteDataManager::get_market_data(const std::string& pair, const Timeframe& timeframe, int limit) {
    std::vector<MarketData> result;
    try {
        pqxx::nontransaction ntxn(*pg_conn_);
        auto res = ntxn.exec_params(
            "SELECT symbol, price, volume, timestamp FROM market_data WHERE symbol = $1 ORDER BY timestamp DESC LIMIT $2",
            pair, limit
        );
        
        for (const auto& row : res) {
            MarketData data;
            data.symbol = row[0].as<std::string>();
            data.price = row[1].as<double>();
            data.volume = row[2].as<double>();
            data.timestamp = row[3].as<std::string>();
            result.push_back(data);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error fetching market data: " << e.what() << std::endl;
        throw;
    }
    return result;
}

void RemoteDataManager::save_training_data(const TrainingData& data) {
    // Stub implementation
    std::cout << "Saving training data for pair: " << data.pair << std::endl;
}

std::vector<TrainingData> RemoteDataManager::get_training_data(const std::string& pair, int limit) {
    // Stub implementation
    std::vector<TrainingData> result;
    std::cout << "Fetching training data for pair: " << pair << ", limit: " << limit << std::endl;
    return result;
}

void RemoteDataManager::cache_model(const std::string& model_id, const std::string& model_data) {
    try {
        redisReply* reply = (redisReply*)redisCommand(redis_conn_, "SET %s %s", model_id.c_str(), model_data.c_str());
        if (!reply) {
            throw std::runtime_error("Redis command failed");
        }
        freeReplyObject(reply);
    } catch (const std::exception& e) {
        std::cerr << "Error caching model: " << e.what() << std::endl;
        throw;
    }
}

std::string RemoteDataManager::get_cached_model(const std::string& model_id) {
    try {
        redisReply* reply = (redisReply*)redisCommand(redis_conn_, "GET %s", model_id.c_str());
        if (!reply) {
            throw std::runtime_error("Redis command failed");
        }
        
        std::string result;
        if (reply->type == REDIS_REPLY_STRING) {
            result = std::string(reply->str, reply->len);
        }
        freeReplyObject(reply);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error getting cached model: " << e.what() << std::endl;
        throw;
    }
}

}
}
