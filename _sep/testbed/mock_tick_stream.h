#pragma once
#include <vector>
#include <functional>
#include <cstdint>
#include "mock_clock.h"

struct MockTick {
    double price;
    uint64_t timestamp; // milliseconds
};

// Deterministic tick stream that feeds ticks to a callback
class MockTickStream {
public:
    using Tick = MockTick;
    using Callback = std::function<void(const Tick&)>;

    explicit MockTickStream(std::vector<Tick> ticks)
        : ticks_(std::move(ticks)) {}

    void run(MockClock& clock, Callback cb) {
        for (const auto& t : ticks_) {
            clock.set(t.timestamp);
            cb(t);
        }
    }
private:
    std::vector<Tick> ticks_;
};
