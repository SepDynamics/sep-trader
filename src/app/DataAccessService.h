#pragma once

#include "IDataAccessService.h"
#include "ServiceBase.h"
#include <hiredis/hiredis.h>
#include <string>

namespace sep {
namespace services {

/**
 * Valkey-backed implementation of IDataAccessService
 */
class DataAccessService : public IDataAccessService, public ServiceBase {
public:
    DataAccessService();
    ~DataAccessService() override;

    bool isReady() const override;

    std::vector<Candle> getHistoricalCandles(
        const std::string& instrument,
        std::uint64_t from,
        std::uint64_t to) override;

protected:
    Result<void> onInitialize() override;
    Result<void> onShutdown() override;

private:
    redisContext* context_;
    std::string host_;
    int port_;
};

} // namespace services
} // namespace sep

