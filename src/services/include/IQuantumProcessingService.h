#pragma once

#include "IService.h"
#include "QuantumTypes.h"
#include <vector>
#include <map>
#include <string>

namespace sep {
namespace services {

/**
 * Interface for the Quantum Processing Service
 * Responsible for quantum algorithms including QBSA (Quantum Binary State Analysis),
 * QFH (Quantum Fourier Hierarchy), pattern evolution, coherence calculation,
 * and stability determination
 */
class IQuantumProcessingService : public IService {
public:
    /**
     * Process a quantum state through the QBSA algorithm
     * @param state Input quantum state
     * @return Result containing processed binary state vector or error
     */
    virtual Result<BinaryStateVector> processBinaryStateAnalysis(const QuantumState& state) = 0;
    
    /**
     * Apply Quantum Fourier Hierarchy analysis to a state
     * @param state Input quantum state
     * @param hierarchyLevels Number of hierarchy levels to compute
     * @return Result containing QFH components or error
     */
    virtual Result<std::vector<QuantumFourierComponent>> applyQuantumFourierHierarchy(
        const QuantumState& state, 
        int hierarchyLevels) = 0;
    
    /**
     * Calculate coherence metrics for a quantum state
     * @param state Input quantum state
     * @return Result containing coherence matrix or error
     */
    virtual Result<CoherenceMatrix> calculateCoherence(const QuantumState& state) = 0;
    
    /**
     * Determine stability metrics for a quantum state
     * @param state Input quantum state
     * @param historicalStates Optional historical states for temporal stability
     * @return Result containing stability metrics or error
     */
    virtual Result<StabilityMetrics> determineStability(
        const QuantumState& state, 
        const std::vector<QuantumState>& historicalStates = {}) = 0;
    
    /**
     * Evolve a quantum state according to pattern evolution rules
     * @param state Input quantum state
     * @param evolutionParameters Parameters controlling the evolution
     * @return Result containing evolved quantum state or error
     */
    virtual Result<QuantumState> evolveQuantumState(
        const QuantumState& state, 
        const std::map<std::string, double>& evolutionParameters) = 0;
    
    /**
     * Run a complete quantum processing pipeline on a state
     * @param state Input quantum state
     * @return Result containing processed quantum state or error
     */
    virtual Result<QuantumState> runQuantumPipeline(const QuantumState& state) = 0;
    
    /**
     * Get available quantum processing algorithms
     * @return Map of algorithm names to descriptions
     */
    virtual std::map<std::string, std::string> getAvailableAlgorithms() const = 0;
};

} // namespace services
} // namespace sep