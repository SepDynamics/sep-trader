#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common/financial_data_types.h"
#include "connectors/oanda_connector.h"

namespace sep
{
    namespace testbed
    {

        inline std::vector<sep::connectors::MarketData> fetchMarketData(
            sep::connectors::OandaConnector& connector, const std::string& pair_symbol,
            size_t hours_back)
        {
            auto now = std::chrono::system_clock::now();
            auto start_time = now - std::chrono::hours(hours_back);

            auto formatTimestamp = [](const std::chrono::system_clock::time_point& tp) {
                auto time_t = std::chrono::system_clock::to_time_t(tp);
                std::stringstream ss;
                ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
                return ss.str();
            };

            std::string from_str = formatTimestamp(start_time);
            std::string to_str = formatTimestamp(now);

            auto oanda_candles = connector.getHistoricalData(pair_symbol, "M1", from_str, to_str);
            if (oanda_candles.empty())
            {
                throw std::runtime_error("No historical data returned from OANDA");
            }

            std::vector<sep::connectors::MarketData> market_data;
            market_data.reserve(oanda_candles.size());

            for (const auto& candle : oanda_candles)
            {
                sep::connectors::MarketData md;
                md.instrument = pair_symbol;
                md.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   sep::common::parseTimestamp(candle.time).time_since_epoch())
                                   .count();
                md.mid = candle.close;
                md.bid = candle.low;
                md.ask = candle.high;
                md.volume = candle.volume;
                md.atr = 0.0;  // placeholder; could compute ATR separately
                market_data.push_back(md);
            }

            return market_data;
        }

    }  // namespace testbed
}  // namespace sep
