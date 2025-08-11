#pragma once

#include <string>
#include <vector>

namespace sep::trading {

class CurrencyQuantumProcessor {
public:
    CurrencyQuantumProcessor() = default;
    std::vector<double> processQuantumSignals(const std::string& pair,
                                              const std::vector<double>& prices);
};

} // namespace sep::trading
