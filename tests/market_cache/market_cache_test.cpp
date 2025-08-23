#include <gtest/gtest.h>
#include "app/market_model_cache.hpp"

using namespace sep;

class DummyPipeline : public IQuantumPipeline {
public:
  int calls = 0;
  bool evaluate_batch(std::span<const SignalProbe> in, std::vector<SignalOut>& out) override {
    ++calls;
    for (const auto& p : in) {
      out.push_back({p.pair + std::to_string(p.t), p.t, p.v, static_cast<uint8_t>(SignalState::Enter)});
    }
    return true;
  }
};

TEST(MarketCache, PipelineCalledOnceAndCached) {
  auto pipeline = std::make_shared<DummyPipeline>();
  apps::MarketModelCache cache(nullptr, pipeline);
  std::vector<Candle> candles(3);
  for (size_t i=0;i<candles.size();++i) {
    candles[i].timestamp = static_cast<time_t>(i+1);
    candles[i].close = 1.0 + i;
  }
  cache.processBatch("EURUSD", candles);
  EXPECT_EQ(pipeline->calls,1);
  EXPECT_EQ(cache.getSignalMap().size(), candles.size());
}

