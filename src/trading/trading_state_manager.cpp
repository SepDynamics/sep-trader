#include "trading_state_manager.hpp"

namespace sep::trading {

void TradingStateManager::setPairState(const std::string& pair, PairState state) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    pair_states_[pair] = state;
}

PairState TradingStateManager::getPairState(const std::string& pair) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = pair_states_.find(pair);
    return it != pair_states_.end() ? it->second : PairState::DISABLED;
}

bool TradingStateManager::isPairEnabled(const std::string& pair) const {
    return getPairState(pair) == PairState::ENABLED;
}

} // namespace sep::trading
