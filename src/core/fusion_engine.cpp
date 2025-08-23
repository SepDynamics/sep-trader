#include "fusion_engine.h"

#include <algorithm>

namespace sep {

SignalOut FusionEngine::fuse(const std::vector<SignalOut>& by_asset, const FusionCfg& c) {
  if (by_asset.empty()) {
    return {"", 0, 0.0, static_cast<uint8_t>(SignalState::None)};
  }
  uint64_t min_t = by_asset.front().t;
  double score = 0.0;
  for (const auto& s : by_asset) {
    min_t = std::min(min_t, s.t);
    auto it = c.weights.find(s.id);
    if (it != c.weights.end()) {
      score += it->second * s.score;
    }
  }
  uint64_t bucket = (c.bucket_ms > 0) ? (min_t / c.bucket_ms) * c.bucket_ms : min_t;
  SignalState state = last_state_;
  if (last_state_ != SignalState::Enter && score > c.enter_hyst) {
    state = SignalState::Enter;
  } else if (last_state_ == SignalState::Enter && score < c.exit_hyst) {
    state = SignalState::Exit;
  } else if (state != SignalState::Enter) {
    state = SignalState::Watch;
  }
  last_state_ = state;
  return {"fusion", bucket, score, static_cast<uint8_t>(state)};
}

} // namespace sep

