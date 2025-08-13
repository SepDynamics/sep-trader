#define SEP_ENABLE_TRACE
#include "../trace.hpp"
#include "../../src/trading/signal_pipeline.hpp"
#include "nlohmann_json_safe.h"
#include <fstream>
#include <vector>
#include <string>

int main() {
    std::ifstream file("assets/test_data/eur_usd_m1_48h.json");
    if (!file.is_open()) return 1;
    nlohmann::json j; file >> j;
    std::vector<double> prices;
    for (const auto& c : j) {
        prices.push_back(c["close"].get<double>());
    }
    auto result = sep::trading::runSignalPipeline("EUR_USD", prices);
    return result.empty();
}
