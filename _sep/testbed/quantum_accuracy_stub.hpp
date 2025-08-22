#pragma once

#include <cmath>
#include <vector>

#include "../../src/connectors/oanda_connector.h"
#include "../../src/trading/quantum_pair_trainer.hpp"

namespace sep {
namespace testbed {

inline double simulate_accuracy(
    const std::vector<sep::connectors::MarketData>& data,
    const sep::trading::QuantumTrainingConfig& config)
{
    (void)data;
    double base_accuracy = 0.58;
    if (std::abs(config.stability_weight - 0.4) < 0.1 &&
        std::abs(config.coherence_weight - 0.1) < 0.05 &&
        std::abs(config.entropy_weight - 0.5) < 0.1)
    {
        base_accuracy += 0.05;
    }
    return base_accuracy;
}

} // namespace testbed
} // namespace sep

