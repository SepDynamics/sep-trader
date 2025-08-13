#pragma once

#include <hiredis/hiredis.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "persistent_pattern_data.hpp"
#include "types.h"

namespace sep::persistence {

// Interface for Redis pattern persistence
class IRedisManager {
public:
    virtual ~IRedisManager() = default;
    virtual void storePattern(std::uint64_t id, const ::sep::persistence::PersistentPatternData& data,
                              const std::string& tier) = 0;
    virtual std::optional<::sep::persistence::PersistentPatternData> loadPattern(
        std::uint64_t id, const std::string& tier) = 0;
    virtual std::vector<std::uint64_t> getPatternIds(const std::string& tier) = 0;
    virtual void removePattern(std::uint64_t id, const std::string& tier) = 0;
    virtual void bulkStore(
        const std::vector<std::pair<std::uint64_t, ::sep::persistence::PersistentPatternData>>& patterns,
        const std::string& tier) = 0;
    virtual std::vector<::sep::persistence::PersistentPatternData> bulkLoad(
        const std::vector<std::uint64_t>& ids, const std::string& tier) = 0;
    virtual void storeHash(const std::string& key, const std::vector<std::pair<std::string, std::string>>& fields) = 0;
    virtual bool isConnected() const = 0;
};

class RedisManager : public ::sep::persistence::IRedisManager {
public:
    RedisManager(const std::string& host, int port);
    ~RedisManager() override;

    void storePattern(std::uint64_t id, const ::sep::persistence::PersistentPatternData& data,
                      const std::string& tier) override;
    std::optional<::sep::persistence::PersistentPatternData> loadPattern(std::uint64_t id,
                                                                  const std::string& tier) override;
    std::vector<std::uint64_t> getPatternIds(const std::string& tier) override;
    void removePattern(std::uint64_t id, const std::string& tier) override;
    void bulkStore(
        const std::vector<std::pair<std::uint64_t, ::sep::persistence::PersistentPatternData>>& patterns,
        const std::string& tier) override;
    std::vector<::sep::persistence::PersistentPatternData> bulkLoad(const std::vector<std::uint64_t>& ids,
                                                             const std::string& tier) override;
    void storeHash(const std::string& key, const std::vector<std::pair<std::string, std::string>>& fields) override;
    bool isConnected() const override;

private:
    class Impl {
    public:
        Impl(const std::string& host, int port);
        ~Impl();
        
        bool isConnected() const;
        void storePattern(std::uint64_t id, const ::sep::persistence::PersistentPatternData& data,
                          const std::string& tier);
        std::optional<::sep::persistence::PersistentPatternData> loadPattern(std::uint64_t id,
                                                                      const std::string& tier);
        std::vector<std::uint64_t> getPatternIds(const std::string& tier);
        void removePattern(std::uint64_t id, const std::string& tier);
        void bulkStore(const std::vector<
                           std::pair<std::uint64_t, ::sep::persistence::PersistentPatternData>>& patterns,
                       const std::string& tier);
        std::vector<::sep::persistence::PersistentPatternData> bulkLoad(
            const std::vector<std::uint64_t>& ids, const std::string& tier);
        void storeHash(const std::string& key, const std::vector<std::pair<std::string, std::string>>& fields);

    private:
        std::string getPatternKey(std::uint64_t id, const std::string& tier) const;
        std::string getTierPatternsKey(const std::string& tier) const;
        std::string normalizeTier(const std::string& tier) const;

        ::redisContext* context_;
        bool connected_;
        std::mutex mutex_;
    };

    std::unique_ptr<Impl> impl_;
};

// Factory function to create RedisManager instances
std::shared_ptr<::sep::persistence::IRedisManager> createRedisManager(const std::string& host = "localhost",
                                                  int port = 6379);

} // namespace sep::persistence
