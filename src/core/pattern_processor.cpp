// Merged from: src/core/bitspace/pattern_processor.cpp
#include "core/pattern_processor.h"

#include <numeric>
#include <cmath>

namespace sep::quantum::bitspace {

PatternProcessor::PatternProcessor(std::map<std::string, std::vector<double>> historical_paths)
    : historical_paths_(std::move(historical_paths)) {}

Metrics PatternProcessor::processTrajectory(Trajectory& trajectory) {
    DampedValue damped_value = trajectory.calculateDampedValue();
    
    Metrics metrics;
    metrics.coherence = calculateCoherence(damped_value);
    metrics.stability = calculateStability(damped_value);
    metrics.entropy = calculateEntropy(damped_value);
    metrics.confidence = matchHistoricalPaths(damped_value.path);
    
    return metrics;
}

double PatternProcessor::calculateCoherence(const DampedValue& damped_value) const {
    if (damped_value.path.size() < 2) {
        return 0.5; // Neutral coherence for single-point path
    }
    // Coherence: 1 - normalized variance of the path
    double mean = std::accumulate(damped_value.path.begin(), damped_value.path.end(), 0.0) / damped_value.path.size();
    double sq_sum = std::inner_product(damped_value.path.begin(), damped_value.path.end(), damped_value.path.begin(), 0.0);
    double variance = (sq_sum / damped_value.path.size()) - (mean * mean);
    return 1.0 - std::min(1.0, variance); // Clamp to [0, 1]
}

double PatternProcessor::calculateStability(const DampedValue& damped_value) const {
    if (!damped_value.converged) {
        return 0.0; // Unstable if not converged
    }
    // Stability: Inverse of the number of steps to converge, normalized
    double stability = 1.0 / static_cast<double>(damped_value.path.size());
    return std::max(0.0, std::min(1.0, stability * 10.0)); // Scale and clamp
}

double PatternProcessor::calculateEntropy(const DampedValue& damped_value) const {
    if (damped_value.path.size() < 2) {
        return 0.0;
    }
    // Entropy: Mean of absolute differences between path points
    double total_diff = 0.0;
    for (size_t i = 1; i < damped_value.path.size(); ++i) {
        total_diff += std::abs(damped_value.path[i] - damped_value.path[i-1]);
    }
    return total_diff / (damped_value.path.size() - 1);
}

double PatternProcessor::matchHistoricalPaths(const std::vector<double>& current_path) const {
    if (historical_paths_.empty()) {
        return 0.5; // Neutral confidence if no history
    }

    double max_similarity = 0.0;

    // Using cosine similarity for path matching
    for (const auto& pair : historical_paths_) {
        const auto& historical_path = pair.second;
        if (current_path.size() != historical_path.size() || current_path.empty()) {
            continue;
        }

        double dot_product = 0.0;
        double norm_current = 0.0;
        double norm_historical = 0.0;

        for (size_t i = 0; i < current_path.size(); ++i) {
            dot_product += current_path[i] * historical_path[i];
            norm_current += current_path[i] * current_path[i];
            norm_historical += historical_path[i] * historical_path[i];
        }

        if (norm_current > 0 && norm_historical > 0) {
            double similarity = dot_product / (std::sqrt(norm_current) * std::sqrt(norm_historical));
            if (similarity > max_similarity) {
                max_similarity = similarity;
            }
        }
    }

    return max_similarity;
}

} // namespace sep::quantum::bitspace

// Merged from: src/core/pattern_processor.cpp
#include "core/common.h"
#include "core/cuda_helpers.h"
#include "core/cuda_sep.h"
#include "core/pattern_types.h"
#include "core/types.h"
#include "core/config.h"
#include "core/pattern_evolution_bridge.h"
#include "core/processor.h"
#include "core/quantum_processor.h"

using ::sep::memory::MemoryTierEnum;

#include <glm/vec3.hpp>

#include <algorithm>
#include <memory>

namespace sep::quantum {
namespace {
class PatternQuantumProcessorImpl final : public sep::pattern::PatternProcessor {
public:
    explicit PatternQuantumProcessorImpl(const QuantumProcessor::Config& config)
        : quantum_processor_(createQuantumProcessor(config)) {}

    sep::ProcessingResult processPattern(
        const sep::quantum::Pattern& pattern) {
        sep::ProcessingResult result;
        result.pattern = pattern;
        result.success = false;
        
        // Convert quantum state to a format the quantum processor can use
        const auto& quantum_state = pattern.quantum_state;
        glm::vec3 stateData(quantum_state.coherence, quantum_state.stability, quantum_state.entropy);
        
        // Process using quantum processor - use uint32_t hash directly
        bool success =
            quantum_processor_->processPattern(stateData, std::hash<uint32_t>{}(pattern.id));

        result.success = success;
        if (success) {
            // Update quantum state values based on processing
            auto& evolved_state = result.pattern.quantum_state;
            evolved_state.coherence = std::min(1.0, quantum_state.coherence * 1.05);
            evolved_state.stability = std::min(1.0, quantum_state.stability * 1.02);
            evolved_state.generation++;

            // Memory tier logic is handled elsewhere - no memory_tier field in QuantumState
            result.success = true;
        } else {
            // Handle error case
            auto& failed_state = result.pattern.quantum_state;
            failed_state.coherence = 0.0f;
            failed_state.stability = 0.0f;
            result.error_message = "QuantumProcessor failed";
            result.success = false;
        }

        return result;
    }

    std::vector<sep::ProcessingResult> processBatch(
        const std::vector<sep::quantum::Pattern>& patterns)
    {
        std::vector<sep::ProcessingResult> results;
        results.reserve(patterns.size());
        
        for (const auto& pattern : patterns) {
            results.push_back(processPattern(pattern));
        }
        
        return results;
    }

    float calculateCoherence(
        const QuantumState& state_a,
        const QuantumState& state_b) const {
        // Convert states to vec3 format for quantum processor
        glm::vec3 vec_a(state_a.coherence, state_a.stability, state_a.entropy);
        glm::vec3 vec_b(state_b.coherence, state_b.stability, state_b.entropy);
        
        // Use quantum processor to calculate coherence
        return quantum_processor_->calculateCoherence(vec_a, vec_b);
    }

    bool isStable(const QuantumState& state) const {
        return state.stability >= sep::quantum::STABILITY_THRESHOLD;
    }

    bool isCollapsed(const QuantumState& state) const {
        return state.coherence < sep::quantum::MIN_COHERENCE;
    }
bool isQuantum(const QuantumState& state) const {
    return state.coherence >= sep::quantum::MIN_COHERENCE;
}
private:
std::unique_ptr<QuantumProcessor> quantum_processor_;
};
}
}

// Function commented out to fix "defined but not used" error
// std::unique_ptr<PatternQuantumProcessorImpl> createPatternQuantumProcessor(
//     const ProcessingConfig& config)
// {
//     QuantumProcessor::Config qp_cfg{};
//     qp_cfg.max_qubits = config.max_patterns;
//     qp_cfg.decoherence_rate = config.mutation_rate;
//     qp_cfg.measurement_threshold = config.ltm_coherence_threshold;
//     qp_cfg.enable_gpu = config.enable_cuda;
//     return std::make_unique<PatternQuantumProcessorImpl>(qp_cfg);
// }

namespace sep::pattern {

PatternProcessor::PatternProcessor(Implementation impl) : implementation_(impl) {}

sep::SEPResult PatternProcessor::init(quantum::GPUContext* ctx) {
    if (implementation_ == Implementation::GPU)
    {
        if (!ctx)
        {
            return sep::SEPResult::INVALID_ARGUMENT;
        }
        CUDA_CHECK(cudaSetDevice(ctx->device_id));
        CUDA_CHECK(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&ctx->default_stream)));
        ctx->initialized = true;
    }
    return sep::SEPResult::SUCCESS;
}

void PatternProcessor::evolvePatterns() {
    for (auto& pattern : patterns_) {
        pattern = mutatePattern(pattern);
    }
}

sep::compat::PatternData PatternProcessor::mutatePattern(const sep::compat::PatternData& parent)
{
    sep::compat::PatternData child = parent;
    // child.generation = parent.generation + 1;
    // child.id = parent.id + "_child";
    return child;
}

sep::SEPResult PatternProcessor::addPattern(const sep::compat::PatternData& pattern)
{
    patterns_.push_back(pattern);
    return sep::SEPResult::SUCCESS;
}

const std::vector<sep::compat::PatternData>& PatternProcessor::getPatterns() const
{
    return patterns_;
}

CPUPatternProcessor::CPUPatternProcessor() : PatternProcessor(Implementation::CPU), patterns_(PatternProcessor::patterns_) {}

sep::SEPResult CPUPatternProcessor::init(quantum::GPUContext* ctx) { return PatternProcessor::init(ctx); }

void CPUPatternProcessor::evolvePatterns() {
    for (auto& p : patterns_) {
        p = mutatePattern(p);
    }
}

sep::compat::PatternData CPUPatternProcessor::mutatePattern(const sep::compat::PatternData& parent)
{
    return PatternProcessor::mutatePattern(parent);
}

} // namespace sep::pattern


// Note: This file was auto-generated and might need manual adjustments
// to align with the project's specific requirements.