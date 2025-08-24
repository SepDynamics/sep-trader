#pragma once

#include "IService.h"
#include "candle_types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sep {
namespace services {

/**
 * Interface for accessing historical market data
 */
class IDataAccessService : public IService {
public:
    virtual ~IDataAccessService() = default;

    /**
     * Retrieve historical candles from Valkey storage
     *
     * @param instrument Instrument symbol (e.g. "EUR_USD")
     * @param from       Start timestamp (inclusive, epoch seconds)
     * @param to         End timestamp (inclusive, epoch seconds)
     * @return Vector of Candle structures ordered by timestamp
     */
    virtual std::vector<Candle> getHistoricalCandles(
        const std::string& instrument,
        std::uint64_t from,
        std::uint64_t to) = 0;
};

} // namespace services
} // namespace sep

