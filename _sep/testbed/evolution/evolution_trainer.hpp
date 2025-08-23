#pragma once
#include "training_session.hpp"
#include "lineage.hpp"

namespace sep::testbed {
class EvolutionTrainer {
public:
    explicit EvolutionTrainer(TrainingSession& session);
    Pattern mutate(const Pattern& parent, double flip_rate, int shift);
    Lineage record_lineage(const Pattern& child,
                           const std::vector<std::string>& parents,
                           const std::string& mutation);
    const std::vector<Lineage>& lineage() const { return lineage_; }
private:
    TrainingSession& session_;
    std::vector<Lineage> lineage_;
};
}
