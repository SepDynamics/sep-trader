#include "pair_optimization_engine.hpp"
#include <numeric>
#include "core/trace.hpp"

namespace sep::trading {

bool PairOptimizationEngine::optimizePair(const std::string& pair, const std::vector<double>& signals) {
    double avg = 0.0;
    if (!signals.empty()) {
        avg = std::accumulate(signals.begin(), signals.end(), 0.0) / signals.size();
    }
    sep::trace::log("decision", pair + " avg=" + std::to_string(avg));
    return avg > 0.5;
}

} // namespace sep::trading
