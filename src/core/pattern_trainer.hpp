// SEP Pattern Trainer Header
// CUDA-accelerated pattern training functionality

#ifndef PATTERN_TRAINER_HPP
#define PATTERN_TRAINER_HPP

#include <string>

namespace sep {
namespace training {

class PatternTrainer {
public:
    PatternTrainer() = default;
    ~PatternTrainer() = default;
    
    bool trainPattern(const std::string& pair);
};

} // namespace training
} // namespace sep

#endif // PATTERN_TRAINER_HPP
