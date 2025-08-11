#pragma once

#include <vector>
#include <complex>
#include <map>
#include <string>

namespace sep {
namespace services {

/**
 * Represents a quantum state in the SEP Engine
 */
struct QuantumState {
    std::vector<std::complex<double>> amplitudes;
    int dimensions;
    double coherenceValue;
    std::string stateIdentifier;
    
    QuantumState() : dimensions(0), coherenceValue(0.0) {}
};

/**
 * Represents a binary state vector derived from quantum state
 */
struct BinaryStateVector {
    std::vector<int> binaryValues;
    double confidence;
    int vectorSize;
    
    BinaryStateVector() : confidence(0.0), vectorSize(0) {}
};

/**
 * Represents a coherence matrix for quantum states
 */
struct CoherenceMatrix {
    std::vector<std::vector<std::complex<double>>> matrix;
    double overallCoherence;
    int matrixDimension;
    
    CoherenceMatrix() : overallCoherence(0.0), matrixDimension(0) {}
};

/**
 * Represents stability metrics for quantum states
 */
struct StabilityMetrics {
    double temporalStability;
    double structuralStability;
    double phaseStability;
    std::map<std::string, double> componentStability;
    
    StabilityMetrics() : temporalStability(0.0), structuralStability(0.0), phaseStability(0.0) {}
};

/**
 * Represents a component in Quantum Fourier Hierarchy
 */
struct QuantumFourierComponent {
    std::vector<std::complex<double>> coefficients;
    int hierarchyLevel;
    double magnitude;
    double phase;
    
    QuantumFourierComponent() : hierarchyLevel(0), magnitude(0.0), phase(0.0) {}
};

} // namespace services
} // namespace sep