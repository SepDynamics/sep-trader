#pragma once
#include "stubs.hpp"
#include "pattern.hpp"
#include <random>
#include <vector>
#include <string>

namespace sep::testbed {
class TrainingSession {
public:
    TrainingSession(sep::engine::EngineFacade& engine,
                    sep::quantum::QFHBasedProcessor& processor,
                    unsigned seed = 42);
    void init();
    double run_epoch(const std::vector<int>& corpus);
    void finalize(const std::string& artifact_path);
    void add_pattern(const Pattern& p);
    bool is_initialized() const { return initialized_; }
    const std::vector<Pattern>& patterns() const { return patterns_; }
    double total_fitness() const;
private:
    sep::engine::EngineFacade& engine_;
    sep::quantum::QFHBasedProcessor& processor_;
    std::vector<Pattern> patterns_;
    std::mt19937 rng_;
    bool initialized_{false};
};
}
