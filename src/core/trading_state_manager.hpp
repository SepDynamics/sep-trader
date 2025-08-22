#pragma once

#include <string>
#include <map>
#include <atomic>
#include <mutex>

namespace sep::trading {

enum class PairState {
    DISABLED,
    TRAINING,
    TRAINED,
    ENABLED,
    ERROR
};

class TradingStateManager {
public:
    TradingStateManager() = default;
    
    void setPairState(const std::string& pair, PairState state);
    PairState getPairState(const std::string& pair) const;
    bool isPairEnabled(const std::string& pair) const;
    
private:
    mutable std::mutex state_mutex_;
    std::map<std::string, PairState> pair_states_;
};

} // namespace sep::trading
