#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>

#include "core/weekly_data_fetcher.hpp"

using namespace sep::train;

TEST(WeeklyDataFetcher, MockResponse)
{
    setenv("OANDA_API_KEY", "dummy", 1);
    setenv("OANDA_ACCOUNT_ID", "dummy", 1);
    setenv("OANDA_ENVIRONMENT", "practice", 1);

    DataFetchConfig cfg;
    cfg.history_days = 7;
    cfg.instruments = {"EUR_USD"};
    cfg.granularities = {"M1"};
    cfg.cache_dir = "test_cache";

    WeeklyDataFetcher fetcher;
    fetcher.configure(cfg);

    namespace fs = std::filesystem;
    fs::path path = fs::path(__FILE__).parent_path() / "fixtures" / "mock_candles.json";
    setenv("OANDA_MOCK_FILE", path.c_str(), 1);

    auto result = fetcher.fetchInstrument("EUR_USD");
    unsetenv("OANDA_MOCK_FILE");

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.candles_fetched, 3u);
    auto hours = std::chrono::duration_cast<std::chrono::hours>(result.end_time - result.start_time).count();
    EXPECT_EQ(hours, cfg.history_days * 24);
    EXPECT_EQ(result.cache_path, cfg.cache_dir + std::string("/EUR_USD_M1.json"));
}
