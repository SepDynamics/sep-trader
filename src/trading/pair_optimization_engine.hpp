#pragma once

#include <string>

namespace sep::trading {

class PairOptimizationEngine {
public:
    PairOptimizationEngine() = default;
    bool optimizePair(const std::string& pair);
};

} // namespace sep::trading
