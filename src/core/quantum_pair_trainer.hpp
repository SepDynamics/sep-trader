/**
 * @file quantum_pair_trainer.hpp
 * @brief Quantum Pair Trainer - Minimal header to avoid STL template issues
 */

#pragma once

// Minimal C-style includes only
#include <cstdint>
#include <cstddef>  // For size_t

// Forward declarations only - no templates
namespace sep {
    namespace engine {
        class EngineFacade;
    }
    
    namespace quantum {
        class QFHBasedProcessor;
        struct QFHOptions;
        struct QFHResult;
        struct Pattern;
        
        namespace manifold {
            class QuantumManifoldOptimizer;
        }
        
        class PatternEvolutionBridge;
    }
    
    namespace connectors {
        struct MarketData;
        class OandaConnector;
    }
    
    namespace persistence {
        class IRedisManager;
    }
    
    namespace trading {
        // Simple POD structures to avoid template issues
        struct PatternDiscoveryResult {
            uint32_t pattern_id;
            double confidence_score;
            double stability_metric;
            uint64_t discovered_timestamp; // Unix timestamp instead of chrono
        };
        
        struct OptimizationResult {
            uint32_t iteration_count;
            double final_score;
            double* parameter_array; // Raw array instead of vector
            size_t parameter_count;
        };
        
        struct TrainingSession {
            uint64_t training_start_timestamp;
            uint64_t training_end_timestamp;
            double accuracy_achieved;
            uint32_t patterns_discovered;
        };
        
        struct PairTrainingResult {
            char pair_symbol[16];  // Fixed size instead of std::string
            double success_score;
            uint32_t pattern_count;
            PatternDiscoveryResult* discovered_patterns; // Raw array
            size_t patterns_size;
            TrainingSession session_info;
            OptimizationResult optimization_details;
        };
        
        /**
         * @class QuantumPairTrainer
         * @brief Minimal implementation avoiding STL templates
         */
        class QuantumPairTrainer {
        public:
            // Constructor/Destructor
            QuantumPairTrainer();
            ~QuantumPairTrainer();
            
            // Core training methods - using C-style interfaces
            PairTrainingResult trainPair(const char* pair_symbol);
            bool trainMultiplePairs(const char** pair_symbols, size_t count, PairTrainingResult* results);
            
            // Configuration
            bool initialize();
            void shutdown();
            bool isInitialized() const;
            
            // Status and monitoring
            bool isTraining() const;
            void stopAllTraining();
            
        private:
            // Raw pointers to avoid template issues
            sep::engine::EngineFacade* engine_facade_;
            sep::quantum::QFHBasedProcessor* qfh_processor_;
            sep::quantum::manifold::QuantumManifoldOptimizer* manifold_optimizer_;
            sep::quantum::PatternEvolutionBridge* pattern_evolver_;
            sep::connectors::OandaConnector* oanda_connector_;
            sep::persistence::IRedisManager* redis_manager_;
            
            // Simple state flags
            bool is_initialized_;
            bool is_training_;
            
        public:
            sep::connectors::MarketData* fetchTrainingData(const char* pair_symbol, size_t* data_count);
            
        private:
            // Private implementation methods
            bool initializeComponents();
            void cleanupComponents();
            uint8_t* convertToBitstream(const sep::connectors::MarketData* data, size_t data_count, size_t* bitstream_size);
            sep::quantum::QFHResult* performQFHAnalysis(const uint8_t* bitstream, size_t bitstream_size);
            sep::quantum::Pattern* discoverPatterns(const sep::connectors::MarketData* data, size_t data_count, size_t* pattern_count);
            OptimizationResult optimizeParameters(const sep::connectors::MarketData* data, size_t data_count);
        };
    }
}