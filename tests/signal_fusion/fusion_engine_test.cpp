#include <gtest/gtest.h>
#include "core/fusion_engine.h"

using namespace sep;

TEST(SignalFusion, PermutationInvariantAndBucket) {
  FusionCfg cfg{1000,0.5,0.3,{{"A",0.6},{"B",0.4}}};
  FusionEngine engine;
  std::vector<SignalOut> v1{{"A",1050,0.6,0},{"B",1090,0.4,0}};
  std::vector<SignalOut> v2{v1[1], v1[0]};
  auto o1 = engine.fuse(v1,cfg);
  auto o2 = engine.fuse(v2,cfg);
  EXPECT_DOUBLE_EQ(o1.score,o2.score);
  EXPECT_EQ(o1.t,o2.t);
}

TEST(SignalFusion, HysteresisBehaviour) {
  FusionCfg cfg{1000,0.5,0.3,{{"A",1.0}}};
  FusionEngine engine;
  std::vector<SignalOut> by{{"A",0,0.6,0}};
  auto o1 = engine.fuse(by,cfg);
  EXPECT_EQ(o1.state, static_cast<uint8_t>(SignalState::Enter));
  by[0].score = 0.4;
  auto o2 = engine.fuse(by,cfg);
  EXPECT_EQ(o2.state, static_cast<uint8_t>(SignalState::Enter));
  by[0].score = 0.2;
  auto o3 = engine.fuse(by,cfg);
  EXPECT_EQ(o3.state, static_cast<uint8_t>(SignalState::Exit));
}

