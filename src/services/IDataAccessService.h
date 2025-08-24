#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "app/candle_types.h"

namespace sep::infra {

using TimePoint = std::chrono::system_clock::time_point;

/**
 * Service interface for accessing historical market data
 */
class IDataAccessService {
public:
    virtual ~IDataAccessService() = default;

    virtual std::vector<Candle> getHistoricalCandles(
        const std::string& instrument,
        TimePoint from,
        TimePoint to) = 0;
};

} // namespace sep::infra

