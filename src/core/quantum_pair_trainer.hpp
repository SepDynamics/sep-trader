/**
 * @file quantum_pair_trainer.hpp
 * @brief Quantum Pair Trainer - Minimal header to avoid STL template issues
 */

#pragma once

// Minimal C-style includes only
#include <cstddef>  // For size_t
#include <cstdint>

struct Candle;

// Forward declarations only - no templates
namespace sep {
namespace quantum {
class QFHBasedProcessor;
struct QFHOptions;
struct QFHResult;

namespace manifold {
class QuantumManifoldOptimizer;
}

class PatternEvolutionBridge;
}  // namespace quantum

namespace connectors {
class OandaConnector;
}  // namespace connectors

namespace persistence {
class IRedisManager;
}

namespace infra {
class IDataAccessService;
}

namespace trading {
struct TrainingSession {
    uint64_t training_start_timestamp;
    uint64_t training_end_timestamp;
    double accuracy_achieved;
    uint32_t patterns_discovered;
};

struct PairTrainingResult {
    char pair_symbol[16];  // Fixed size instead of std::string
    TrainingSession session_info;
};

/**
 * @class QuantumPairTrainer
 * @brief Minimal implementation avoiding STL templates
 */
class QuantumPairTrainer {
  public:
    // Constructor/Destructor
    explicit QuantumPairTrainer(sep::infra::IDataAccessService* data_access);
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
    sep::quantum::QFHBasedProcessor* qfh_processor_;
    sep::quantum::manifold::QuantumManifoldOptimizer* manifold_optimizer_;
    sep::quantum::PatternEvolutionBridge* pattern_evolver_;
    sep::connectors::OandaConnector* oanda_connector_;
    sep::persistence::IRedisManager* redis_manager_;
    sep::infra::IDataAccessService* data_access_;

    // Simple state flags
    bool is_initialized_;
    bool is_training_;

  private:
    // Private implementation methods
    bool initializeComponents();
    void cleanupComponents();
    uint8_t* convertToBitstream(const ::Candle* data, size_t data_count, size_t* bitstream_size);
    sep::quantum::QFHResult* performQFHAnalysis(const uint8_t* bitstream, size_t bitstream_size);
};
}  // namespace trading
}  // namespace sep
