#pragma once

#include "ServiceBase.h"
#include "candle_types.h"
#include <hiredis/hiredis.h>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

namespace sep {
namespace services {

/**
 * Valkey-backed service for historical market data
 */
class DataAccessService : public ServiceBase {
public:
    DataAccessService();
    ~DataAccessService() override;

    bool isReady() const override;

    std::vector<Candle> getHistoricalCandles(
        const std::string& instrument,
        std::uint64_t from,
        std::uint64_t to);

protected:
    Result<void> onInitialize() override;
    Result<void> onShutdown() override;

private:
    static std::pair<std::string, int> parseValkeyUrl(const std::string& url);

    redisContext* context_;
    std::string host_;
    int port_;
};

} // namespace services
} // namespace sep

