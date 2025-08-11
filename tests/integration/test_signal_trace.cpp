#include "trading/ticker_pattern_analyzer.hpp"
#include "connectors/oanda_connector.h"
#include <gtest/gtest.h>
#include <vector>

using namespace sep;

TEST(TradingIntegration, EmitsBuySignalForUptrend) {
    trading::PatternAnalysisConfig config;
    trading::TickerPatternAnalyzer analyzer(config);

    std::vector<connectors::MarketData> data;
    for (int i = 0; i < 5; ++i) {
        connectors::MarketData md;
        md.instrument = "EUR_USD";
        md.mid = 1.0 + 0.01 * i;
        md.bid = md.mid - 0.0001;
        md.ask = md.mid + 0.0001;
        md.timestamp = static_cast<uint64_t>(i);
        md.volume = 100;
        data.push_back(md);
    }

    auto analysis = analyzer.analyzeFromMarketData("EUR_USD", data);
    EXPECT_EQ(analysis.primary_signal, trading::TickerPatternAnalysis::SignalDirection::BUY);
}
