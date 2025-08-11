#include "signal_pipeline.hpp"
#include "currency_quantum_processor.hpp"
#include "pair_optimization_engine.hpp"
#include "../../_sep/testbed/trace.hpp"

namespace sep::trading {
std::string runSignalPipeline(const std::string& pair, const std::vector<double>& prices) {
    sep::testbed::trace("fetch", std::to_string(prices.size()) + " prices");
    CurrencyQuantumProcessor processor;
    auto signals = processor.processQuantumSignals(pair, prices);
    PairOptimizationEngine engine;
    bool decision = engine.optimizePair(pair, signals);
    return decision ? "trade" : "hold";
}
}
