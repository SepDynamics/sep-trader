#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include "io/oanda_connector.h"

namespace sep {
namespace tests {

namespace {
std::time_t parseTime(const std::string& ts) {
    std::string trimmed = ts;
    auto z_pos = trimmed.find('Z');
    if (z_pos != std::string::npos) {
        trimmed = trimmed.substr(0, z_pos);
    }
    auto dot_pos = trimmed.find('.');
    if (dot_pos != std::string::npos) {
        trimmed = trimmed.substr(0, dot_pos);
    }
    std::tm tm{};
    std::istringstream ss(trimmed);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        return -1;
    }
#if defined(_WIN32) || defined(_WIN64)
    return _mkgmtime(&tm);
#else
    return timegm(&tm);
#endif
}
} // namespace

TEST(OandaConnectorIntegration, FetchesRecentHistoricalData) {
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

    auto to_iso = [](auto tp) {
        std::time_t tt = std::chrono::system_clock::to_time_t(tp);
        std::tm tm = *gmtime(&tt);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
        return oss.str();
    };

    std::string from = to_iso(from_tp);
    std::string to = to_iso(to_tp);

    auto candles = connector.getHistoricalData("EUR_USD", "M1", from, to);
    ASSERT_FALSE(candles.empty());

    auto first_ts = parseTime(candles.front().time);
    auto last_ts = parseTime(candles.back().time);
    ASSERT_NE(first_ts, -1);
    ASSERT_NE(last_ts, -1);

    std::time_t from_t = std::chrono::system_clock::to_time_t(from_tp);
    std::time_t to_t = std::chrono::system_clock::to_time_t(to_tp);

    EXPECT_GE(first_ts, from_t);
    EXPECT_LE(last_ts, to_t + 60); // allow small slack
}

} // namespace tests
} // namespace sep

