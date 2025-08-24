#include "core/sep_precompiled.h"
#include "pattern_evolution_trainer.hpp"
#include "core/pattern_evolution_bridge.h"
#include "core/processor.h"
#include "core/engine.h"
#include "util/memory_tier_manager.hpp"
#include "core/logging.h"
#include <memory>
#include <vector>

namespace sep::trading {

bool PatternEvolutionTrainer::evolvePatterns(const std::string& pair) {
    try {
        LOG_INFO("Starting pattern evolution for trading pair: " + pair);
        
        // Initialize pattern evolution bridge with optimal quantum parameters
        sep::quantum::PatternEvolutionBridge::Config evolution_config;
        evolution_config.entanglement_threshold = 0.6f;         // Higher threshold for trading patterns
        evolution_config.collapse_variance_threshold = 0.25f;   // Lower threshold for market volatility
        evolution_config.environment_coupling = 0.02f;          // Market environment coupling
        evolution_config.target_coherence = 0.85f;              // High coherence target for reliable signals
        evolution_config.target_stability = 0.75f;              // Stability for risk management
        evolution_config.evolution_step_size = 0.03f;           // Conservative evolution steps
        evolution_config.coupling_strength = 0.5f;              // Medium coupling for balanced evolution
        
        auto evolution_bridge = std::make_unique<sep::quantum::PatternEvolutionBridge>(evolution_config);
        
        // Retrieve patterns from memory tiers for the specified trading pair
        std::vector<sep::quantum::Pattern> patterns_to_evolve;
        
        // Access patterns from STM first (most recent trading data)
        auto memory_manager = sep::memory::MemoryTierManager::getInstance();
        if (memory_manager) {
            auto stm_patterns = memory_manager->getPatternsForPair(pair, sep::memory::MemoryTierEnum::STM);
            patterns_to_evolve.insert(patterns_to_evolve.end(), stm_patterns.begin(), stm_patterns.end());
            
            // Also get patterns from MTM for historical context
            auto mtm_patterns = memory_manager->getPatternsForPair(pair, sep::memory::MemoryTierEnum::MTM);
            patterns_to_evolve.insert(patterns_to_evolve.end(), mtm_patterns.begin(), mtm_patterns.end());
            
            LOG_INFO("Retrieved " + std::to_string(patterns_to_evolve.size()) + " patterns for evolution");
        }
        
        // If no patterns found in memory tiers, get from processor
        if (patterns_to_evolve.empty()) {
            LOG_WARNING("No patterns found in memory tiers, accessing processor patterns");
            
            // Access patterns from quantum processor
            auto processor = sep::core::Engine::getInstance()->getQuantumProcessor();
            if (processor) {
                auto processor_patterns = processor->getPatternsForPair(pair);
                patterns_to_evolve = processor_patterns;
            }
        }
        
        if (patterns_to_evolve.empty()) {
            LOG_WARNING("No patterns available for evolution for pair: " + pair);
            return false;
        }
        
        // Initialize evolution state
        evolution_bridge->initializeEvolutionState();
        
        // Perform quantum pattern evolution with adaptive time step
        float evolution_time_step = 0.1f;  // Base time step for market data
        auto evolution_result = evolution_bridge->evolvePatterns(patterns_to_evolve, evolution_time_step);
        
        // Validate evolution results
        if (evolution_result.evolved_patterns.empty()) {
            LOG_ERROR("Pattern evolution failed - no evolved patterns produced");
            return false;
        }
        
        // Check for quantum coherence improvements
        if (evolution_result.total_coherence < 0.4f) {
            LOG_WARNING("Low coherence after evolution: " + std::to_string(evolution_result.total_coherence));
        }
        
        // Detect and handle quantum collapse events
        auto collapse_event = evolution_bridge->detectCollapse(evolution_result.evolved_patterns);
        if (collapse_event.detected) {
            LOG_WARNING("Quantum collapse detected during evolution - severity: " +
                       std::to_string(collapse_event.severity));
            
            // Apply collapse recovery for critical patterns
            if (collapse_event.severity > 0.8f) {
                evolution_bridge->updatePatterns(evolution_result.evolved_patterns);
            }
        }
        
        // Compute entanglements for pattern relationships
        auto entanglements = evolution_bridge->computeEntanglements(evolution_result.evolved_patterns);
        LOG_INFO("Found " + std::to_string(entanglements.size()) + " quantum entanglements");
        
        // Update patterns with evolved quantum states
        evolution_bridge->updatePatterns(evolution_result.evolved_patterns);
        
        // Store evolved patterns back to appropriate memory tiers
        if (memory_manager) {
            for (const auto& evolved_pattern : evolution_result.evolved_patterns) {
                // Promote high-coherence patterns to higher memory tiers
                if (evolved_pattern.quantum_state.coherence > evolution_config.target_coherence) {
                    memory_manager->storePattern(evolved_pattern, sep::memory::MemoryTierEnum::MTM);
                } else {
                    memory_manager->storePattern(evolved_pattern, sep::memory::MemoryTierEnum::STM);
                }
            }
        }
        
        // Log evolution metrics
        LOG_INFO("Pattern evolution completed successfully for " + pair +
                ": coherence=" + std::to_string(evolution_result.total_coherence) +
                ", entropy_change=" + std::to_string(evolution_result.entropy_change) +
                ", stability=" + std::to_string(evolution_result.stability_metric));
        
        // Success if we have meaningful coherence improvement
        return evolution_result.total_coherence > 0.3f && evolution_result.stability_metric > 0.2f;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Pattern evolution failed for pair " + pair + ": " + e.what());
        return false;
    }
}

} // namespace sep::trading
