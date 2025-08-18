#pragma once

#include <complex>
#include <string>
#include <vector>

#include "core/types.h"

namespace sep::quantum::manifold {

enum class ManifoldQuantumState {
    SUPERPOSITION,
    COHERENT,
    COLLAPSED
};

struct QuantumPattern {
    std::string id;
    std::vector<double> position;
    double coherence;
    double stability;
    int generation;
    ::sep::quantum::manifold::ManifoldQuantumState state;
    double phase;
    std::complex<double> amplitude;
};

} // namespace sep::quantum::manifold