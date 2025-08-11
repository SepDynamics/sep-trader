#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

// Simple moving average helper
std::vector<double> computeSMA(const std::vector<double>& data, std::size_t period)
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

// Generate simple buy/sell/hold signals based on price vs SMA
std::vector<std::string> generateSignals(const std::vector<double>& prices, const std::vector<double>& sma)
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

int main()
{
    // Ingest historical OANDA data from assets
    std::ifstream file("../assets/test_data/eur_usd_m1_48h.json");
    if (!file.is_open())
    {
        std::cerr << "Failed to open data file\n";
        return 1;
    }
    nlohmann::json j;
    file >> j;
    std::vector<double> prices;
    for (const auto& c : j)
        prices.push_back(c["close"].get<double>());

    auto sma = computeSMA(prices, 3);
    auto signals = generateSignals(prices, sma);

    // Write resulting signals to testbed for inspection
    nlohmann::json out;
    out["signals"] = signals;
    std::ofstream outFile("signals.json");
    outFile << out.dump(2);
    return 0;
}
