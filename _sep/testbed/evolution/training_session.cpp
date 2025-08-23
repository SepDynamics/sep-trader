#include "training_session.hpp"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace sep::testbed {
TrainingSession::TrainingSession(sep::engine::EngineFacade& engine,
                                 sep::quantum::QFHBasedProcessor& processor,
                                 unsigned seed)
    : engine_(engine), processor_(processor), rng_(seed) {}

void TrainingSession::init() {
    initialized_ = true;
}

double TrainingSession::run_epoch(const std::vector<int>& corpus) {
    if (!initialized_) return 0.0;
    if (patterns_.empty()) {
        for (size_t i = 0; i < corpus.size(); ++i) {
            Pattern p;
            p.id = "p" + std::to_string(i);
            p.mask.push_back(static_cast<uint64_t>(corpus[i]));
            patterns_.push_back(p);
        }
    }
    std::uniform_real_distribution<double> dist(0.0, 0.1);
    double total = 0.0;
    for (auto& p : patterns_) {
        engine_.process(p.mask);
        processor_.analyze(p.mask);
        double delta = dist(rng_);
        p.fitness += delta;
        total += delta;
    }
    return total;
}

void TrainingSession::finalize(const std::string& artifact_path) {
    if (!initialized_) return;
    std::ofstream out(artifact_path);
    for (const auto& p : patterns_) {
        out << p.id << "," << p.fitness << "\n";
    }
    out.close();
    patterns_.clear();
    initialized_ = false;
}

void TrainingSession::add_pattern(const Pattern& p) {
    patterns_.push_back(p);
}

double TrainingSession::total_fitness() const {
    double sum = 0.0;
    for (const auto& p : patterns_) sum += p.fitness;
    return sum;
}
}
