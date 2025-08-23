#include "evolution_trainer.hpp"
#include <random>

namespace sep::testbed {
EvolutionTrainer::EvolutionTrainer(TrainingSession& session)
    : session_(session) {}

Pattern EvolutionTrainer::mutate(const Pattern& parent, double flip_rate, int shift) {
    Pattern child = parent;
    child.id = parent.id + "_m";
    std::mt19937 rng(0);
    std::bernoulli_distribution flip(flip_rate);
    for (auto& m : child.mask) {
        if (flip(rng)) {
            m ^= 1ULL; // flip least significant bit
        }
        if (shift > 0)
            m <<= shift;
        else if (shift < 0)
            m >>= -shift;
    }
    return child;
}

Lineage EvolutionTrainer::record_lineage(const Pattern& child,
                                         const std::vector<std::string>& parents,
                                         const std::string& mutation) {
    Lineage l;
    l.id = child.id;
    l.generation = static_cast<int>(lineage_.size()) + 1;
    l.parents = parents;
    l.mutation = mutation;
    lineage_.push_back(l);
    return l;
}
}
