#pragma once

#include "ServiceBase.h"
#include "IQuantumProcessingService.h"
#include <memory>
#include <unordered_map>

namespace sep {
namespace services {

/**
 * Implementation of the Quantum Processing Service
 * Provides concrete implementations of quantum algorithms and processing
 */
// Fix multiple inheritance issue by inheriting only from ServiceBase and implementing IQuantumProcessingService
class QuantumProcessingService : public ServiceBase, public IQuantumProcessingService {
public:
    /**
     * Constructor
     */
    QuantumProcessingService();
    
    /**
     * Destructor
     */
    virtual ~QuantumProcessingService() = default;
    
    // Override isReady() to resolve the diamond inheritance issue
    bool isReady() const override;
    
    // IQuantumProcessingService interface implementation
    Result<BinaryStateVector> processBinaryStateAnalysis(const QuantumState& state) override;
    Result<std::vector<QuantumFourierComponent>> applyQuantumFourierHierarchy(
        const QuantumState& state,
        int hierarchyLevels) override;
    Result<CoherenceMatrix> calculateCoherence(const QuantumState& state) override;
    Result<StabilityMetrics> determineStability(
        const QuantumState& state,
        const std::vector<QuantumState>& historicalStates) override;
    Result<QuantumState> evolveQuantumState(
        const QuantumState& state,
        const std::map<std::string, double>& evolutionParameters) override;
    Result<QuantumState> runQuantumPipeline(const QuantumState& state) override;
    std::map<std::string, std::string> getAvailableAlgorithms() const override;

protected:
    // ServiceBase interface implementation
    Result<void> onInitialize() override;
    Result<void> onShutdown() override;

private:
    // Algorithm implementations
    BinaryStateVector performQBSA(const QuantumState& state);
    std::vector<QuantumFourierComponent> performQFH(const QuantumState& state, int levels);
    CoherenceMatrix computeCoherenceMatrix(const QuantumState& state);
    StabilityMetrics computeStabilityMetrics(const QuantumState& state, const std::vector<QuantumState>& history);
    QuantumState performEvolution(const QuantumState& state, const std::map<std::string, double>& params);
    
    // Cache for expensive computations
    std::unordered_map<std::string, BinaryStateVector> qbsaCache_;
    std::unordered_map<std::string, CoherenceMatrix> coherenceCache_;
    
    // Available algorithms mapping
    std::map<std::string, std::string> algorithms_;
};

} // namespace services
} // namespace sep