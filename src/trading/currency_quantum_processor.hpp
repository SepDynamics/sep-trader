#pragma once

#include <string>
#include <vector>

namespace sep::trading {

class CurrencyQuantumProcessor {
public:
    CurrencyQuantumProcessor() = default;
    std::vector<double> processQuantumSignals(const std::string& pair);
};

} // namespace sep::trading
