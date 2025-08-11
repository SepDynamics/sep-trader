#pragma once

#include <string>
#include <vector>

namespace sep::trading {

class PairOptimizationEngine {
public:
    PairOptimizationEngine() = default;
    bool optimizePair(const std::string& pair, const std::vector<double>& signals);
};

} // namespace sep::trading
