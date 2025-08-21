#pragma once

#include "app/ServiceBase.h"
#include "IQuantumProcessingService.h"
#include "core/result_types.h"
#include "core/qfh.h"
#include "core/quantum_processor_qfh.h"
#include "core/quantum_types.h"
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

// Forward declarations from core quantum namespace
namespace sep {
namespace quantum {
class QuantumProcessorQFH;
class QFH;
struct QFHResult;
}
}

namespace sep {
namespace services {

/**
 * Implementation of the Quantum Processing Service
 * Provides concrete implementations of quantum algorithms and processing
 */
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
    std::map<std::string, std::string> getAvailableAlgorithms() const;

protected:
    // ServiceBase interface implementation
    Result<void> onInitialize() override;
    Result<void> onShutdown() override;

private:
    // Authentic algorithm implementations - Bit-Transition Harmonic Analysis
    BinaryStateVector performAuthenticQBSA(const QuantumState& state);
    std::vector<QuantumFourierComponent> performAuthenticQFH(const QuantumState& state, int levels);
    CoherenceMatrix computeAuthenticCoherenceMatrix(const QuantumState& state);
    StabilityMetrics computeAuthenticStabilityMetrics(const QuantumState& state, const std::vector<QuantumState>& history);
    QuantumState performAuthenticEvolution(const QuantumState& state, const std::map<std::string, double>& params);
    
    // Cache for expensive computations
    std::unordered_map<std::string, BinaryStateVector> qbsaCache_;
    std::unordered_map<std::string, CoherenceMatrix> coherenceCache_;
    
    // Available algorithms mapping
    std::unordered_map<std::string, std::string> algorithms_;
    
    // Authentic quantum processor instances
    std::unique_ptr<::sep::quantum::QuantumProcessorQFH> qfh_processor_;
    std::unique_ptr<::sep::quantum::QFHBasedProcessor> qfh_engine_;
    
    // Type conversion helpers between service layer and core quantum layer
    glm::vec3 convertToGLMPattern(const QuantumState& serviceState);
    QuantumState convertFromGLMPattern(const glm::vec3& pattern, const std::string& identifier);
    BinaryStateVector convertFromBitVector(const std::vector<std::uint32_t>& bits);
    std::vector<QuantumFourierComponent> convertFromQFHResult(const ::sep::quantum::QFHResult& qfhResult);
    CoherenceMatrix buildCoherenceMatrixFromStability(double stability);
    StabilityMetrics buildStabilityFromProcessing(double stability, double coherence);
};

} // namespace services
} // namespace sep