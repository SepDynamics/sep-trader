#include <gtest/gtest.h>
#include "core/signal_types.h"

using namespace sep;

TEST(PatternMetricEngine, EvaluateSignalCases) {
  RuleParams p{0.35,1000,0.6};
  std::vector<std::pair<RuleInput, SignalState>> cases = {
    {{0.36,0,1000,0,0.7}, SignalState::Enter},
    {{0.36,0,1000,1,0.7}, SignalState::Exit},
    {{0.34,0,1000,0,0.7}, SignalState::Watch},
    {{0.36,0,900,0,0.7}, SignalState::None},
    {{0.36,0,1000,0,0.5}, SignalState::Watch},
    {{0.35,0,1000,0,0.7}, SignalState::Watch},
    {{0.36,0,-1,0,0.7}, SignalState::None},
    {{std::numeric_limits<double>::quiet_NaN(),0,1000,0,0.7}, SignalState::None}
  };
  for (const auto& [in, expected] : cases) {
    EXPECT_EQ(evaluateSignal(in,p), expected);
  }
}

