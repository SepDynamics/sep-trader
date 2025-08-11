#pragma once
#include <functional>
#include <algorithm>
#include <cstring>
#include "mock_tick_stream.h"
#include "core_types/candle_data.h"

// Assembles ticks into fixed-period candles
class CandleAssembler {
public:
    using Tick = MockTickStream::Tick;
    using Candle = sep::core::CandleData;
    using Callback = std::function<void(const Candle&)>;

    CandleAssembler(uint64_t period_ms, Callback cb)
        : period_ms_(period_ms), callback_(std::move(cb)) {}

    void onTick(const Tick& tick) {
        if (!has_candle_) {
            startNewCandle(tick);
            return;
        }
        if (tick.timestamp >= candle_start_ + period_ms_) {
            finalize();
            startNewCandle(tick);
        } else {
            updateCandle(tick);
        }
    }

    void finalize() {
        if (has_candle_) {
            callback_(current_);
            has_candle_ = false;
        }
    }

private:
    void startNewCandle(const Tick& tick) {
        candle_start_ = tick.timestamp - (tick.timestamp % period_ms_);
        current_.timestamp = candle_start_;
        current_.open = current_.high = current_.low = current_.close = tick.price;
        current_.volume = 1;
        current_.period = static_cast<uint32_t>(period_ms_ / 1000);
        std::strncpy(current_.symbol, "TEST", sizeof(current_.symbol));
        has_candle_ = true;
    }

    void updateCandle(const Tick& tick) {
        current_.high = std::max(current_.high, tick.price);
        current_.low = std::min(current_.low, tick.price);
        current_.close = tick.price;
        current_.volume += 1;
    }

    uint64_t period_ms_;
    Callback callback_;
    uint64_t candle_start_{0};
    Candle current_{};
    bool has_candle_{false};
};
