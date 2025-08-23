#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "app/multi_asset_signal_fusion.hpp"
#include "app/enhanced_market_model_cache.hpp"
#include "app/quantum_signal_bridge.hpp"
#include "app/candle_types.h"

#include <memory>

// Mock for QuantumSignalBridge
class MockQuantumSignalBridge : public sep::trading::QuantumSignalBridge {
public:
    // No methods need to be mocked for this test
};

// Mock for EnhancedMarketModelCache
class MockEnhancedMarketModelCache : public sep::cache::EnhancedMarketModelCache {
public:
    MockEnhancedMarketModelCache() : sep::cache::EnhancedMarketModelCache(nullptr, nullptr) {}
    MOCK_METHOD(std::vector<Candle>, getRecentCandles, (const std::string& pair, int count), ());
};

class MultiAssetFusionTest : public ::testing::Test {
protected:
    void SetUp() override {
        quantum_processor = std::make_shared<MockQuantumSignalBridge>();
        market_cache = std::make_shared<testing::StrictMock<MockEnhancedMarketModelCache>>();
        fusion = std::make_unique<sep::MultiAssetSignalFusion>(quantum_processor, market_cache);
    }

    std::shared_ptr<MockQuantumSignalBridge> quantum_processor;
    std::shared_ptr<MockEnhancedMarketModelCache> market_cache;
    std::unique_ptr<sep::MultiAssetSignalFusion> fusion;
};


TEST_F(MultiAssetFusionTest, PositiveCorrelation) {
    // Feed known correlated sequences
    std::vector<Candle> candles1, candles2;
    for (int i = 0; i < 100; ++i) {
        double price = 1.0 + i * 0.01;
        candles1.push_back({.close = price, .time = std::to_string(i)});
        candles2.push_back({.close = price, .time = std::to_string(i)});
    }

    EXPECT_CALL(*market_cache, getRecentCandles("EUR_USD", 100)).WillOnce(testing::Return(candles1));
    EXPECT_CALL(*market_cache, getRecentCandles("USD_JPY", 100)).WillOnce(testing::Return(candles2));

    auto correlation = fusion->calculateDynamicCorrelation("EUR_USD", "USD_JPY");
    
    // Verify correlation > 0.7
    EXPECT_GT(correlation.strength, 0.99);
}

TEST_F(MultiAssetFusionTest, NegativeCorrelation) {
    // Feed inverse sequences
    std::vector<Candle> candles1, candles2;
    for (int i = 0; i < 100; ++i) {
        candles1.push_back({.close = 1.0 + i * 0.01, .time = std::to_string(i)});
        candles2.push_back({.close = 1.0 - i * 0.01, .time = std::to_string(i)});
    }

    EXPECT_CALL(*market_cache, getRecentCandles("EUR_USD", 100)).WillOnce(testing::Return(candles1));
    EXPECT_CALL(*market_cache, getRecentCandles("USD_JPY", 100)).WillOnce(testing::Return(candles2));

    auto correlation = fusion->calculateDynamicCorrelation("EUR_USD", "USD_JPY");

    // Verify correlation < -0.7
    EXPECT_LT(correlation.strength, -0.99);
}

TEST_F(MultiAssetFusionTest, ZeroCorrelation) {
    // Feed random uncorrelated data
    std::vector<Candle> candles1, candles2;
    srand(0);
    for (int i = 0; i < 100; ++i) {
        candles1.push_back({.close = static_cast<double>(rand()) / RAND_MAX, .time = std::to_string(i)});
        candles2.push_back({.close = static_cast<double>(rand()) / RAND_MAX, .time = std::to_string(i)});
    }

    EXPECT_CALL(*market_cache, getRecentCandles("EUR_USD", 100)).WillOnce(testing::Return(candles1));
    EXPECT_CALL(*market_cache, getRecentCandles("USD_JPY", 100)).WillOnce(testing::Return(candles2));

    auto correlation = fusion->calculateDynamicCorrelation("EUR_USD", "USD_JPY");

    // Verify correlation near 0.0
    EXPECT_NEAR(correlation.strength, 0.0, 0.2);
}
