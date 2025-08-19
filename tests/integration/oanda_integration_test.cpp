#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <vector>
#include "io/oanda_connector.h"

// This integration test requires OANDA sandbox credentials provided via
// OANDA_API_KEY and OANDA_ACCOUNT_ID environment variables.

namespace {
std::string toIso(const std::chrono::system_clock::time_point& tp) {
    std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm tm = *gmtime(&tt);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

bool isUniform(const std::vector<sep::connectors::OandaCandle>& candles) {
    if (candles.size() < 2) {
        return true;
    }
    const auto& first = candles.front();
    for (const auto& c : candles) {
        if (c.open != first.open || c.high != first.high || c.low != first.low || c.close != first.close) {
            return false;
        }
    }
    return true;
}
} // namespace

TEST(OandaIntegration, SandboxHistoricalDataIsReal) {
    const char* api_key = std::getenv("OANDA_API_KEY");
    const char* account_id = std::getenv("OANDA_ACCOUNT_ID");
    if (!api_key || !account_id) {
        GTEST_SKIP() << "OANDA credentials not set";
    }

    sep::connectors::OandaConnector connector(api_key, account_id, true);
    ASSERT_TRUE(connector.initialize()) << connector.getLastError();

    auto now = std::chrono::system_clock::now();
    auto to_tp = std::chrono::time_point_cast<std::chrono::seconds>(now);
    auto from_tp = to_tp - std::chrono::hours(1);

    std::string from = toIso(from_tp);
    std::string to = toIso(to_tp);

    auto candles = connector.getHistoricalData("EUR_USD", "M1", from, to);
    ASSERT_FALSE(candles.empty()) << "No candles returned";
    EXPECT_FALSE(isUniform(candles)) << "Received uniform candles - possible mock data";
}
