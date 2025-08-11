#pragma once
#include <cstdint>

// Simple deterministic mock clock for testing
class MockClock {
public:
    uint64_t now() const { return current_; }
    void set(uint64_t t) { current_ = t; }
private:
    uint64_t current_{0};
};
