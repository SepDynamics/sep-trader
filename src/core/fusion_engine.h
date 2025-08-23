#pragma once

#include "core/signal_types.h"

namespace sep {

class FusionEngine {
public:
  SignalOut fuse(const std::vector<SignalOut>& by_asset, const FusionCfg& c);
private:
  SignalState last_state_ = SignalState::None;
};

} // namespace sep

