#include "core/sep_precompiled.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

using sep::connectors::OandaConnector;

namespace {
std::string formatTimestamp(const std::chrono::system_clock::time_point &tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}
} // namespace

int main() {
    const char *api_key = std::getenv("OANDA_API_KEY");
    const char *account_id = std::getenv("OANDA_ACCOUNT_ID");
    if (!api_key || !account_id) {
        std::cerr << "OANDA_API_KEY and OANDA_ACCOUNT_ID must be set in environment." << std::endl;
        return 1;
    }

    OandaConnector connector(api_key, account_id, true);
    if (!connector.initialize()) {
        std::cerr << "Failed to initialize OandaConnector: " << connector.getLastError() << std::endl;
        return 1;
    }

    auto now = std::chrono::system_clock::now();
    auto start = now - std::chrono::hours(48);
    auto from = formatTimestamp(start);
    auto to = formatTimestamp(now);

    auto candles = connector.getHistoricalData("EUR_USD", "M1", from, to);
    if (candles.empty()) {
        std::cerr << "Failed to fetch historical data: " << connector.getLastError() << std::endl;
        connector.shutdown();
        return 1;
    }

    std::cout << "Fetched " << candles.size() << " candles for EUR_USD." << std::endl;

    connector.shutdown();
    return 0;
}

