#include "services/IDataAccessService.h"
#include "services/ServiceLocator.h"
#include "util/nlohmann_json_safe.h"

#include <hiredis/hiredis.h>
#include <chrono>
#include <fstream>

namespace sep::infra {

/**
 * Lightweight Valkey-backed data access service used by lower-level modules.
 */
class DataAccessService : public IDataAccessService {
public:
    DataAccessService() {
        // Load connection parameters
        std::string host = "127.0.0.1";
        int port = 6379;
        std::ifstream f("config/default.json");
        if (f) {
            nlohmann::json j; f >> j;
            if (j.contains("valkey")) {
                host = j["valkey"].value("host", host);
                port = j["valkey"].value("port", port);
            }
        }
        ctx_ = redisConnect(host.c_str(), port);
        ServiceLocator::provide(std::shared_ptr<IDataAccessService>(this, [](IDataAccessService*){}));
    }

    ~DataAccessService() override {
        if (ctx_) redisFree(ctx_);
    }

    std::vector<Candle> getHistoricalCandles(
        const std::string& instrument, TimePoint from, TimePoint to) override {
        std::vector<Candle> candles;
        if (!ctx_) return candles;

        auto from_s = std::chrono::duration_cast<std::chrono::seconds>(from.time_since_epoch()).count();
        auto to_s   = std::chrono::duration_cast<std::chrono::seconds>(to.time_since_epoch()).count();

        std::string key = "md:price:" + instrument;
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(ctx_, "ZRANGEBYSCORE %s %lld %lld", key.c_str(),
                         static_cast<long long>(from_s), static_cast<long long>(to_s)));
        if (!reply) return candles;

        for (size_t i = 0; i < reply->elements; ++i) {
            const char* json_str = reply->element[i]->str;
            if (!json_str) continue;
            try {
                auto j = nlohmann::json::parse(json_str);
                Candle c{};
                c.timestamp = j.value("t", 0ull);
                c.open = std::stod(j.value("o", "0"));
                c.high = std::stod(j.value("h", "0"));
                c.low = std::stod(j.value("l", "0"));
                c.close = std::stod(j.value("c", "0"));
                c.volume = j.value("v", 0.0);
                candles.push_back(c);
            } catch (...) {
                // ignore malformed
            }
        }
        freeReplyObject(reply);
        return candles;
    }

private:
    redisContext* ctx_{nullptr};
};

} // namespace sep::infra

