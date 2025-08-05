#pragma once

#include <string>
#include <vector>
#include <memory>
#include <pqxx/pqxx>
#include <hiredis/hiredis.h>

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
};

enum class Timeframe {
    M1, M5, M15, M30, H1, H4, D1
};

class RemoteDataManager {
public:
    RemoteDataManager(const std::string& postgres_conn_str, const std::string& redis_url);
    ~RemoteDataManager();

    // Market data synchronization
    void save_market_data(const MarketData& data);
    std::vector<MarketData> get_market_data(const std::string& pair, const Timeframe& timeframe, int limit);

    // Training data management
    void save_training_data(const TrainingData& data);
    std::vector<TrainingData> get_training_data(const std::string& pair, int limit);

    // Model cache operations
    void cache_model(const std::string& model_id, const std::string& model_data);
    std::string get_cached_model(const std::string& model_id);

private:
    std::unique_ptr<pqxx::connection> pg_conn_;
    redisContext* redis_conn_;
};

}
}
