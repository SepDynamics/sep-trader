#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>
#include "util/nlohmann_json_safe.h"

static std::vector<double> ingestPrices(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        return {};
    nlohmann::json j;
    file >> j;
    std::vector<double> prices;
    for (const auto& c : j)
        prices.push_back(c["close"].get<double>());
    return prices;
}

static std::vector<double> computeSMA(const std::vector<double>& data, std::size_t period)
{
    std::vector<double> sma(data.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
    {
        sum += data[i];
        if (i >= period)
            sum -= data[i - period];
        std::size_t denom = std::min(i + 1, period);
        sma[i] = sum / static_cast<double>(denom);
    }
    return sma;
}

static std::vector<std::string> generateSignals(const std::vector<double>& prices, const std::vector<double>& sma)
{
    std::vector<std::string> signals(prices.size());
    for (std::size_t i = 0; i < prices.size(); ++i)
    {
        if (prices[i] > sma[i])
            signals[i] = "BUY";
        else if (prices[i] < sma[i])
            signals[i] = "SELL";
        else
            signals[i] = "HOLD";
    }
    return signals;
}

TEST(DataPipeline, Stage1Ingestion)
{
    auto prices = ingestPrices("assets/test_data/eur_usd_m1_48h.json");
    ASSERT_GE(prices.size(), 5u);
    EXPECT_NEAR(prices[0], 1.17717, 1e-5);
}

TEST(DataPipeline, Stage2Processing)
{
    std::vector<double> sample{1.17717, 1.17716, 1.17718, 1.17719, 1.17715};
    auto sma = computeSMA(sample, 3);
    ASSERT_EQ(sma.size(), sample.size());
    EXPECT_NEAR(sma[0], 1.17717, 1e-5);
    EXPECT_NEAR(sma[1], 1.177165, 1e-6);
    EXPECT_NEAR(sma[2], 1.17717, 1e-5);
    EXPECT_NEAR(sma[3], 1.1771766667, 1e-6);
    EXPECT_NEAR(sma[4], 1.1771733333, 1e-6);
}

TEST(DataPipeline, Stage3SignalGeneration)
{
    std::vector<double> sample{1.17717, 1.17716, 1.17718, 1.17719, 1.17715};
    auto sma = computeSMA(sample, 3);
    auto signals = generateSignals(sample, sma);
    std::vector<std::string> expected{"HOLD", "SELL", "BUY", "BUY", "SELL"};
    EXPECT_EQ(signals, expected);
}
