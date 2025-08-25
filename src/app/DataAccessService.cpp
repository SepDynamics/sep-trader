#include "DataAccessService.h"
#include "util/nlohmann_json_safe.h"

#include <fstream>
#include <cstdlib>

namespace sep {
namespace services {

std::pair<std::string, int> DataAccessService::parseValkeyUrl(const std::string& url) {
    std::string host = "localhost";
    int port = 6379;
    if (url.rfind("redis://", 0) == 0) {
        std::string rest = url.substr(8);
        size_t colon = rest.find(':');
        if (colon != std::string::npos) {
            host = rest.substr(0, colon);
            try {
                port = std::stoi(rest.substr(colon + 1));
            } catch (...) {
                port = 6379;
            }
        } else {
            host = rest;
        }
    }
    return {host, port};
}

DataAccessService::DataAccessService()
    : ServiceBase("DataAccessService", "1.0.0"),
      context_(nullptr), host_("127.0.0.1"), port_(6379) {}

DataAccessService::~DataAccessService() {
    onShutdown();
}

bool DataAccessService::isReady() const {
    return context_ != nullptr;
}

Result<void> DataAccessService::onInitialize() {
    const char* url_env = std::getenv("VALKEY_URL");
    if (url_env) {
        auto parsed = parseValkeyUrl(url_env);
        host_ = parsed.first;
        port_ = parsed.second;
    } else {
        // Load Valkey connection parameters from config/default.json
        std::ifstream f("config/default.json");
        if (f) {
            nlohmann::json j; f >> j;
            if (j.contains("valkey")) {
                host_ = j["valkey"].value("host", host_);
                port_ = j["valkey"].value("port", port_);
            }
        }
    }

    context_ = redisConnect(host_.c_str(), port_);
    if (!context_ || context_->err) {
        std::string err = context_ ? context_->errstr : "unknown";
        if (context_) {
            redisFree(context_);
            context_ = nullptr;
        }
        return Error(Error::Code::ResourceUnavailable,
                     "Failed to connect to Valkey: " + err);
    }

    return {};
}

Result<void> DataAccessService::onShutdown() {
    if (context_) {
        redisFree(context_);
        context_ = nullptr;
    }
    return {};
}

std::vector<Candle> DataAccessService::getHistoricalCandles(
    const std::string& instrument, std::uint64_t from, std::uint64_t to) {
    std::vector<Candle> candles;
    if (!context_) return candles;

    std::string key = "md:price:" + instrument;
    redisReply* reply = static_cast<redisReply*>(
        redisCommand(context_, "ZRANGEBYSCORE %s %llu %llu", key.c_str(),
                     static_cast<unsigned long long>(from),
                     static_cast<unsigned long long>(to)));
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
            // ignore malformed entry
        }
    }
    freeReplyObject(reply);

    return candles;
}

} // namespace services
} // namespace sep

