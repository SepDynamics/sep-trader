#pragma once

#include <string>

namespace sep::trading {

class PatternEvolutionTrainer {
public:
    PatternEvolutionTrainer() = default;
    bool evolvePatterns(const std::string& pair);
};

} // namespace sep::trading
