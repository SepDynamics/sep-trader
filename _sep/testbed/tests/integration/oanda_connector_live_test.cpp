#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include "io/oanda_connector.h"

// Live OANDA connector test using real API calls.
// Only runs when OANDA_LIVE_TEST=1 is set to avoid accidental execution.

TEST(OandaConnectorLiveTest, AccountAndInstrumentResponsesAreNonEmpty) {
    const char* live_flag = std::getenv("OANDA_LIVE_TEST");
    if (!live_flag || std::string(live_flag) != "1") {
        GTEST_SKIP() << "OANDA_LIVE_TEST not set";
    }

    const char* api_key = std::getenv("OANDA_API_KEY");
    const char* account_id = std::getenv("OANDA_ACCOUNT_ID");
    if (!api_key || !account_id) {
        GTEST_SKIP() << "OANDA credentials not set";
    }

    bool sandbox = true;
    if (const char* env = std::getenv("OANDA_ENVIRONMENT"); env && std::string(env) == "live") {
        sandbox = false;
    }

    sep::connectors::OandaConnector connector(api_key, account_id, sandbox);
    ASSERT_TRUE(connector.initialize()) << connector.getLastError();

    auto account = connector.getAccountInfo();
    ASSERT_FALSE(account.empty()) << "Account info response empty";

    auto instruments = connector.getInstruments();
    ASSERT_FALSE(instruments.empty()) << "Instruments response empty";
}

