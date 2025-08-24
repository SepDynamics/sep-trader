#include "market_model_cache.hpp"

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include "util/nlohmann_json_safe.h"
#include <thread>
#include <cstring>

#include "core/sep_precompiled.h"

namespace sep::apps {

MarketModelCache::MarketModelCache(std::shared_ptr<sep::connectors::OandaConnector> connector,
                                   std::shared_ptr<IQuantumPipeline> pipeline)
    : oanda_connector_(std::move(connector)), pipeline_(std::move(pipeline)), valkey_context_(nullptr) {
    // Get Valkey URL from environment, defaulting to localhost
    valkey_url_ = std::getenv("VALKEY_URL") ? std::getenv("VALKEY_URL") : "redis://localhost:6379";
    connectToValkey();
}

const std::map<std::string, sep::trading::QuantumTradingSignal>& MarketModelCache::getSignalMap() const {
    return processed_signals_;
}

bool MarketModelCache::ensureCacheForLastWeek(const std::string& instrument) {
    std::string cache_key = getCacheKeyForLastWeek(instrument);

    if (!valkey_context_) {
        std::cout << "[CACHE] ❌ No Valkey connection available" << std::endl;
        return false;
    }

    // Check if cache exists in Valkey
    redisReply* reply = (redisReply*)redisCommand(valkey_context_, "EXISTS %s", cache_key.c_str());
    bool cache_exists = (reply && reply->integer == 1);
    if (reply) freeReplyObject(reply);

    if (cache_exists) {
        std::cout << "[CACHE] ✅ Found existing cache in Valkey. Loading..." << std::endl;
        return loadCacheFromValkey(cache_key);
    }

    std::cout << "[CACHE] 🔄 No cache found in Valkey. Fetching fresh data for the last trading week..." << std::endl;

    // Instead of trying to fetch old data, get the most recent available data
    // OANDA will give us the latest candles automatically with count parameter
    
    std::vector<Candle> raw_candles;
    bool data_fetched = false;
    
    std::cout << "[CACHE] 📥 Requesting most recent 2880 M1 candles (48 hours of trading data)" << std::endl;
    
    // Use the existing OANDA connector API with empty from/to to get latest data
    auto oanda_candles = oanda_connector_->getHistoricalData(instrument, "M1", "", "");
    std::cout << "[CACHE] 📊 Received " << oanda_candles.size() << " candles from OANDA" << std::endl;
    
    // Convert OandaCandle to local Candle format
    for (const auto& o_candle : oanda_candles) {
        Candle c;
        c.timestamp = parseTimestamp(o_candle.time);
        c.open = o_candle.open;
        c.high = o_candle.high;
        c.low = o_candle.low;
        c.close = o_candle.close;
        c.volume = static_cast<double>(o_candle.volume);
        raw_candles.push_back(c);
    }
    data_fetched = true;

    // Wait for async fetch to complete (with timeout)
    int timeout_seconds = 30;
    while (!data_fetched && timeout_seconds-- > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (timeout_seconds % 5 == 0) {
            std::cout << "[CACHE] ⏳ Still waiting for OANDA response... (" << timeout_seconds << "s remaining)" << std::endl;
        }
    }
    
    if (!data_fetched || raw_candles.empty()) {
        std::cerr << "[CACHE] ❌ OANDA data not available and cache miss in Valkey" << std::endl;
        return false;
    }

    std::cout << "[CACHE] ⚡ Processing " << raw_candles.size() << " candles through quantum pipeline..." << std::endl;
    processAndCacheData(raw_candles, cache_key);
    return true;
}

void MarketModelCache::processBatch(const std::string& instrument, const std::vector<Candle>& candles) {
    processed_signals_.clear();
    if (!pipeline_) return;
    std::vector<SignalProbe> probes;
    probes.reserve(candles.size());
    for (const auto& c : candles) {
        probes.push_back(SignalProbe{instrument, static_cast<uint64_t>(c.timestamp), c.close});
    }
    std::vector<SignalOut> outs;
    if (pipeline_->evaluate_batch(probes, outs)) {
        for (const auto& out : outs) {
            sep::trading::QuantumTradingSignal s;
            if (out.state == static_cast<uint8_t>(SignalState::Enter)) {
                s.action = sep::trading::QuantumTradingSignal::BUY;
            } else if (out.state == static_cast<uint8_t>(SignalState::Exit)) {
                s.action = sep::trading::QuantumTradingSignal::SELL;
            } else {
                s.action = sep::trading::QuantumTradingSignal::HOLD;
            }
            s.identifiers.confidence = static_cast<float>(std::clamp(out.score, 0.0, 1.0));
            s.identifiers.coherence = 0.0f;
            s.identifiers.stability = 0.0f;
            processed_signals_[out.id] = s;
        }
    }
}

void MarketModelCache::processAndCacheData(const std::vector<Candle>& raw_candles, const std::string& cache_key) {
    processBatch("EUR_USD", raw_candles);
    std::cout << "[CACHE] ✅ Processing complete. Generated " << processed_signals_.size() << " signals." << std::endl;
    saveCacheToValkey(cache_key);
}

bool MarketModelCache::saveCacheToValkey(const std::string& cache_key) const {
    if (!valkey_context_) {
        std::cerr << "[CACHE] ❌ No Valkey connection available for saving" << std::endl;
        return false;
    }
    
    nlohmann::json j;
    j["metadata"]["instrument"] = "EUR_USD";
    j["metadata"]["created_at"] = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    j["metadata"]["signal_count"] = processed_signals_.size();
    
    for (const auto& [timestamp, signal] : processed_signals_) {
        nlohmann::json signal_json;
        signal_json["action"] = (signal.action == sep::trading::QuantumTradingSignal::BUY) ? "BUY" :
                               (signal.action == sep::trading::QuantumTradingSignal::SELL) ? "SELL" : "HOLD";
        signal_json["confidence"] = signal.identifiers.confidence;
        signal_json["coherence"] = signal.identifiers.coherence;
        signal_json["stability"] = signal.identifiers.stability;
        
        j["signals"][timestamp] = signal_json;
    }
    
    std::string json_data = j.dump();
    
    // Store in Valkey using the canonical cache key pattern
    redisReply* reply = (redisReply*)redisCommand(valkey_context_, "SET %s %s", cache_key.c_str(), json_data.c_str());
    
    bool success = (reply && reply->type == REDIS_REPLY_STATUS && strcmp(reply->str, "OK") == 0);
    if (reply) freeReplyObject(reply);
    
    if (success) {
        // Set expiration to 7 days (604800 seconds)
        reply = (redisReply*)redisCommand(valkey_context_, "EXPIRE %s %d", cache_key.c_str(), 604800);
        if (reply) freeReplyObject(reply);
        
        std::cout << "[CACHE] 💾 Saved " << processed_signals_.size() << " signals to Valkey key: " << cache_key << std::endl;
        return true;
    } else {
        std::cerr << "[CACHE] ❌ Failed to save cache to Valkey: " << cache_key << std::endl;
        return false;
    }
}

bool MarketModelCache::loadCacheFromValkey(const std::string& cache_key) {
    if (!valkey_context_) {
        std::cerr << "[CACHE] ❌ No Valkey connection available for loading" << std::endl;
        return false;
    }
    
    redisReply* reply = (redisReply*)redisCommand(valkey_context_, "GET %s", cache_key.c_str());
    
    if (!reply || reply->type != REDIS_REPLY_STRING) {
        if (reply) freeReplyObject(reply);
        std::cerr << "[CACHE] ❌ Failed to get cache from Valkey: " << cache_key << std::endl;
        return false;
    }
    
    std::string json_data(reply->str, reply->len);
    freeReplyObject(reply);
    
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_data);
    } catch (const std::exception& e) {
        std::cerr << "[CACHE] ❌ Failed to parse cache JSON from Valkey: " << e.what() << std::endl;
        return false;
    }

    processed_signals_.clear();
    
    if (j.contains("signals")) {
        for (auto const& [timestamp, signal_json] : j["signals"].items()) {
            sep::trading::QuantumTradingSignal signal;
            
            std::string action = signal_json["action"];
            if (action == "BUY") {
                signal.action = sep::trading::QuantumTradingSignal::BUY;
            } else if (action == "SELL") {
                signal.action = sep::trading::QuantumTradingSignal::SELL;
            } else {
                signal.action = sep::trading::QuantumTradingSignal::HOLD;
            }
            
            signal.identifiers.confidence = signal_json.value("confidence", 0.5f);
            signal.identifiers.coherence = signal_json.value("coherence", 0.5f);
            signal.identifiers.stability = signal_json.value("stability", 0.5f);
            
            processed_signals_[timestamp] = signal;
        }
    }

    std::cout << "[CACHE] 📊 Loaded " << processed_signals_.size() << " signals from Valkey cache." << std::endl;
    
    if (j.contains("metadata")) {
        std::cout << "[CACHE] 📅 Cache created: " << j["metadata"].value("created_at", 0) << std::endl;
        std::cout << "[CACHE] 📈 Original signal count: " << j["metadata"].value("signal_count", 0) << std::endl;
    }
    
    return true;
}

std::string MarketModelCache::getCacheKeyForLastWeek(const std::string& instrument) const {
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    std::tm* gmt = std::gmtime(&now_t);
    
    char week_str[16];
    std::strftime(week_str, sizeof(week_str), "%Y-W%U", gmt); // Format as Year-WeekNumber
    
    // Use canonical cache key pattern: cache:signals:{instrument}:{week}
    return "cache:signals:" + instrument + ":" + week_str;
}

bool MarketModelCache::connectToValkey() {
    if (valkey_context_) {
        disconnectFromValkey();
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
    
    valkey_context_ = redisConnect(host.c_str(), port);
    
    if (!valkey_context_ || valkey_context_->err) {
        std::cerr << "[CACHE] ❌ Failed to connect to Valkey at " << host << ":" << port;
        if (valkey_context_) {
            std::cerr << " - " << valkey_context_->errstr;
            redisFree(valkey_context_);
            valkey_context_ = nullptr;
        }
        std::cerr << std::endl;
        return false;
    }
    
    std::cout << "[CACHE] 🔗 Connected to Valkey at " << host << ":" << port << std::endl;
    return true;
}

void MarketModelCache::disconnectFromValkey() {
    if (valkey_context_) {
        redisFree(valkey_context_);
        valkey_context_ = nullptr;
    }
}

} // namespace sep::apps
