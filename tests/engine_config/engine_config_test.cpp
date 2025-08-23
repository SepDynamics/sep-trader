#include <gtest/gtest.h>
#include "core/engine_config_io.h"

using namespace sep;

TEST(EngineConfig, RoundTrip) {
  EngineConfig cfg;
  cfg.rules = {0.35,1000,0.6};
  cfg.fusion.bucket_ms = 1000;
  cfg.fusion.enter_hyst = 0.05;
  cfg.fusion.exit_hyst = 0.03;
  cfg.fusion.weights = {{"EURUSD",0.4},{"GBPUSD",0.3},{"USDJPY",0.3}};
  cfg.error_templates = {{"CONFIG_MISSING","Required field '{field}' is missing"},
                         {"OUT_OF_RANGE","Field '{field}' out of range: {value}"}};
  auto json = save_engine_config(cfg);
  EngineConfig loaded;
  ASSERT_TRUE(load_engine_config(json, loaded));
  auto json2 = save_engine_config(loaded);
  EXPECT_EQ(json, json2);
  EXPECT_EQ(loaded.fusion.weights.at("USDJPY"), 0.3);
}

TEST(EngineConfig, MissingRulesFails) {
  std::string j = R"({"version":1,"fusion":{"bucket_ms":1,"enter_hyst":0.1,"exit_hyst":0.1,"weights":{"A":1}},"error_templates":{}})";
  EngineConfig cfg;
  EXPECT_FALSE(load_engine_config(j, cfg));
}

