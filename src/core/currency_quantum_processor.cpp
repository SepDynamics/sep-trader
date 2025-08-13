#include "currency_quantum_processor.hpp"
#include "trace.hpp"
#include <vector>

namespace sep::trading {

std::vector<double> CurrencyQuantumProcessor::processQuantumSignals(const std::string& pair,
                                              const std::vector<double>& prices) {
    sep::testbed::trace("process", pair + " " + std::to_string(prices.size()) + " pts");
    std::vector<double> signals;
    signals.reserve(prices.size());
    for (double p : prices) {
        signals.push_back(p / 100.0);
    }
    if (!signals.empty())
        sep::testbed::trace("signal", "first=" + std::to_string(signals.front()));
    return signals;
}

} // namespace sep::trading
