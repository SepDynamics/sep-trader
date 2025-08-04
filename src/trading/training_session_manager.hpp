#pragma once

#include <string>

namespace sep::trading {

class TrainingSessionManager {
public:
    TrainingSessionManager() = default;
    bool startTrainingSession(const std::string& pair);
    void endTrainingSession(const std::string& pair);
};

} // namespace sep::trading
