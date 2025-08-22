#include "connectors/oanda_connector.h"
#include <cstdlib>
#include <iostream>

int main() {
    const char* api_key = std::getenv("OANDA_API_KEY");
    const char* account_id = std::getenv("OANDA_ACCOUNT_ID");
    if (!api_key || !account_id) {
        std::cerr << "Missing OANDA credentials" << std::endl;
        return 1;
    }
    sep::connectors::OandaConnector connector(api_key, account_id, true);
    if (!connector.initialize()) {
        std::cerr << "Init failed: " << connector.getLastError() << std::endl;
        return 1;
    }
    auto candles = connector.getHistoricalData("EUR_USD", "M1", "", "");
    if (candles.empty()) {
        std::cerr << "No data returned" << std::endl;
        return 1;
    }
    std::cout << "Received " << candles.size() << " candles" << std::endl;

    // Intentionally request an invalid instrument to verify error handling
    auto bad = connector.getHistoricalData("BAD_PAIR", "M1", "", "");
    if (bad.empty()) {
        std::cout << "Invalid instrument correctly returned no data" << std::endl;
    } else {
        std::cerr << "Unexpected data for invalid instrument" << std::endl;
    }
    return 0;
}
