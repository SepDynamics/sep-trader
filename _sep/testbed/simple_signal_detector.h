#pragma once
#include "core_types/candle_data.h"

// Records if an upward closing candle occurred
class SimpleSignalDetector {
public:
    void onCandle(const sep::core::CandleData& c) {
        if (c.close > c.open) {
            up_ = true;
        }
    }
    bool upTriggered() const { return up_; }
private:
    bool up_{false};
};
