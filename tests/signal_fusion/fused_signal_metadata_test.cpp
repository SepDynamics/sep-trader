#include <gtest/gtest.h>
#include "app/multi_asset_signal_fusion.hpp"

using namespace sep;

TEST(FusedSignalMetadata, SerializeIncludesFields) {
    MultiAssetSignalFusion fusion(nullptr, nullptr);
    FusedSignal signal{Direction::BUY, 0.8, {}, 0.9, 0.7, "abc123", "1.2"};
    auto json = fusion.serializeFusionResult(signal);
    EXPECT_NE(json.find("\"input_hash\": \"abc123\""), std::string::npos);
    EXPECT_NE(json.find("\"config_version\": \"1.2\""), std::string::npos);
}
