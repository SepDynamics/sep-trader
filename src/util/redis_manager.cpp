#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>

#include "core/logging.h"
#include "core/types.h"
#include "util/redis_manager.h"
#include "util/memory_tier_manager.hpp"

#include <hiredis/hiredis.h>

// Define namespace alias to clarify that Manager is in the logging namespace
namespace logging = sep::logging;

namespace sep::persistence {

    std::string RedisManager::Impl::getPatternKey(std::uint64_t id, const std::string& tier) const
    {
        std::stringstream key;
        key << "pattern:" << normalizeTier(tier) << ":" << id;
        return key.str();
    }

    std::string RedisManager::Impl::getTierPatternsKey(const std::string& tier) const
    {
        std::stringstream key;
        key << normalizeTier(tier) << ":patterns";
        return key.str();
    }
std::string RedisManager::Impl::normalizeTier(const std::string& tier) const
{
    std::string lower_tier = tier;
    std::transform(lower_tier.begin(), lower_tier.end(), lower_tier.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return lower_tier;
}

// RedisManager::Impl method implementations

RedisManager::Impl::Impl(const std::string& host, int port) : context_(nullptr), connected_(false)
{
    auto logger = logging::Manager::getInstance().getLogger("redis");
    context_ = redisConnect(host.c_str(), port);
    if (context_ == nullptr || context_->err) {
        if (logger) {
            if (context_) {
                logger->error("Redis connection error: {}", context_->errstr);
            } else {
                logger->error("Redis connection error: cannot allocate redis context");
            }
        }
    } else {
        connected_ = true;
        if (logger) {
            logger->info("Redis connection established");
        }
    }
}

RedisManager::Impl::~Impl() {
    if (context_) {
        redisFree(context_);
        context_ = nullptr;
    }
}

bool RedisManager::Impl::isConnected() const { return connected_; }

void RedisManager::Impl::storePattern(std::uint64_t id, const PersistentPatternData& data,
                                      const std::string& tier)
{
    if (!connected_ || !context_)
        return;

    auto logger = logging::Manager::getInstance().getLogger("redis");
    std::lock_guard<std::mutex> lock(mutex_);

    auto pattern_key = getPatternKey(id, tier);
    auto patterns_key = getTierPatternsKey(tier);

    redisReply* reply = static_cast<redisReply*>(redisCommand(context_, "DEL %s", pattern_key.c_str()));
    if (reply)
        freeReplyObject(reply);

    reply = static_cast<redisReply*>(redisCommand(
        context_,
        "HSET %s coherence %f stability %f generation_count %d",
        pattern_key.c_str(),
        static_cast<double>(data.coherence),
        static_cast<double>(data.stability),
        data.generation_count));
    if (reply)
        freeReplyObject(reply);

    reply = static_cast<redisReply*>(redisCommand(context_, "SADD %s %zu", patterns_key.c_str(), id));
    if (reply)
        freeReplyObject(reply);

    if (logger) {
        logger->debug("Stored pattern {} in tier {}", id, tier.c_str());
    }
}

std::optional<PersistentPatternData> RedisManager::Impl::loadPattern(std::uint64_t id,
                                                                     const std::string& tier)
{
    if (!connected_ || !context_)
        return std::nullopt;

    std::lock_guard<std::mutex> lock(mutex_);

    auto pattern_key = getPatternKey(id, tier);

    redisReply* reply = static_cast<redisReply*>(redisCommand(context_, "EXISTS %s", pattern_key.c_str()));
    if (!reply || reply->type != REDIS_REPLY_INTEGER || reply->integer == 0) {
        if (reply)
            freeReplyObject(reply);
        return std::nullopt;
    }
    freeReplyObject(reply);

    PersistentPatternData pattern_data{};

    // Get coherence
    reply = static_cast<redisReply*>(redisCommand(context_, "HGET %s coherence", pattern_key.c_str()));
    if (reply && reply->type == REDIS_REPLY_STRING)
        pattern_data.coherence = std::stof(reply->str);
    if (reply)
        freeReplyObject(reply);

    // Get stability
    reply = static_cast<redisReply*>(redisCommand(context_, "HGET %s stability", pattern_key.c_str()));
    if (reply && reply->type == REDIS_REPLY_STRING)
        pattern_data.stability = std::stof(reply->str);
    if (reply)
        freeReplyObject(reply);

    // Get generation count
    reply = static_cast<redisReply*>(redisCommand(context_, "HGET %s generation_count", pattern_key.c_str()));
    if (reply && reply->type == REDIS_REPLY_STRING)
        pattern_data.generation_count = std::stoi(reply->str);
    if (reply)
        freeReplyObject(reply);

    return pattern_data;
}

std::vector<std::uint64_t> RedisManager::Impl::getPatternIds(const std::string& tier)
{
    std::vector<std::uint64_t> ids;
    if (!connected_ || !context_)
        return ids;

    std::lock_guard<std::mutex> lock(mutex_);

    auto patterns_key = getTierPatternsKey(tier);
    redisReply* reply = static_cast<redisReply*>(redisCommand(context_, "SMEMBERS %s", patterns_key.c_str()));
    if (reply && reply->type == REDIS_REPLY_ARRAY) {
        ids.reserve(reply->elements);
        for (size_t i = 0; i < reply->elements; ++i) {
            if (reply->element[i]->str)
                ids.push_back(std::stoull(reply->element[i]->str));
        }
    }
    if (reply)
        freeReplyObject(reply);

    return ids;
}

void RedisManager::Impl::removePattern(std::uint64_t id, const std::string& tier)
{
    if (!connected_ || !context_)
        return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto pattern_key = getPatternKey(id, tier);
    auto patterns_key = getTierPatternsKey(tier);

    redisReply* reply = static_cast<redisReply*>(redisCommand(context_, "SREM %s %zu", patterns_key.c_str(), id));
    if (reply)
        freeReplyObject(reply);
    reply = static_cast<redisReply*>(redisCommand(context_, "DEL %s", pattern_key.c_str()));
    if (reply)
        freeReplyObject(reply);
}

void RedisManager::Impl::bulkStore(
    const std::vector<std::pair<std::uint64_t, PersistentPatternData>>& patterns,
    const std::string& tier)
{
    for (const auto& p : patterns) {
        storePattern(p.first, p.second, tier);
    }
}

std::vector<PersistentPatternData> RedisManager::Impl::bulkLoad(
    const std::vector<std::uint64_t>& ids, const std::string& tier)
{
    std::vector<PersistentPatternData> results;
    results.reserve(ids.size());
    for (std::uint64_t id : ids) {
        auto data = loadPattern(id, tier);
        if (data)
            results.push_back(*data);
    }
    return results;
}

void RedisManager::Impl::storeHash(const std::string& key, const std::vector<std::pair<std::string, std::string>>& fields)
{
    if (!connected_ || !context_)
        return;

    auto logger = logging::Manager::getInstance().getLogger("redis");
    std::lock_guard<std::mutex> lock(mutex_);

    // Construct the HSET command with multiple field-value pairs
    std::vector<const char*> argv;
    argv.push_back("HSET");
    argv.push_back(key.c_str());
    for (const auto& field : fields) {
        argv.push_back(field.first.c_str());
        argv.push_back(field.second.c_str());
    }

    redisReply* reply = static_cast<redisReply*>(redisCommandArgv(context_, argv.size(), argv.data(), NULL));
    if (reply) {
        freeReplyObject(reply);
    }

    if (logger) {
        logger->debug("Stored hash {} with {} fields", key, fields.size());
    }
}

// RedisManager implementation
RedisManager::RedisManager(const std::string& host, int port)
    : impl_(std::make_unique<Impl>(host, port))
{
}
RedisManager::~RedisManager() = default;
std::shared_ptr<IRedisManager> createRedisManager(const std::string& host, int port)
{
    return std::make_shared<RedisManager>(host, port);
}
void RedisManager::storePattern(std::uint64_t id,
                                const sep::persistence::PersistentPatternData& data,
                                const std::string& tier)
{
    impl_->storePattern(id, data, tier);
}

std::optional<sep::persistence::PersistentPatternData> RedisManager::loadPattern(
    std::uint64_t id, const std::string& tier)
{
    return impl_->loadPattern(id, tier);
}

std::vector<std::uint64_t> RedisManager::getPatternIds(const std::string& tier)
{
    return impl_->getPatternIds(tier);
}

void RedisManager::removePattern(std::uint64_t id, const std::string& tier)
{
    impl_->removePattern(id, tier);
}

void RedisManager::bulkStore(
    const std::vector<std::pair<std::uint64_t, sep::persistence::PersistentPatternData>>& patterns,
    const std::string& tier)
{
    impl_->bulkStore(patterns, tier);
}

std::vector<sep::persistence::PersistentPatternData> RedisManager::bulkLoad(
    const std::vector<std::uint64_t>& ids, const std::string& tier)
{
    return impl_->bulkLoad(ids, tier);
}

void RedisManager::storeHash(const std::string& key, const std::vector<std::pair<std::string, std::string>>& fields)
{
    impl_->storeHash(key, fields);
}

bool RedisManager::isConnected() const
{
    return impl_->isConnected();
}

}  // namespace sep::persistence