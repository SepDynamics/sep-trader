#pragma once
#include "common/sep_precompiled.h"
#include "connectors/oanda_connector.h"
#include "core_types/result.h"
#include "engine/facade/facade.h"
#include "quantum/bitspace/qfh.h"
#include "quantum/pattern_evolution_bridge.h"
#include "quantum/quantum_manifold_optimizer.h"

namespace sep
{
    namespace persistence
    {
        class IRedisManager;
    }
    namespace memory
    {
        using IRedisManager = sep::persistence::IRedisManager;
    }
} // namespace sep

namespace sep::trading
{

    /**
     * Quantum Training Configuration for Currency Pairs
     * Based on 60.73% accuracy breakthrough parameters
     */
    struct QuantumTrainingConfig
    {
        // Optimal weights from breakthrough analysis
        double stability_weight = 0.4;  // 40% weight with inversion logic
        double coherence_weight = 0.1;  // 10% minimal influence
        double entropy_weight = 0.5;    // 50% primary signal driver

        // Optimal thresholds from systematic optimization
        double confidence_threshold = 0.65;  // High-confidence threshold
        double coherence_threshold = 0.30;   // Coherence threshold

        // Training parameters
        size_t training_window_hours = 120;  // 5 days of M1 data
        size_t pattern_analysis_depth = 50;  // Forward window size
        size_t max_training_iterations = 1000;
        double convergence_tolerance = 1e-6;

        // Multi-timeframe analysis
        bool enable_m5_analysis = true;
        bool enable_m15_analysis = true;
        bool require_triple_confirmation = true;

        // CUDA optimization
        bool enable_cuda_acceleration = true;
        size_t cuda_batch_size = 256;
        size_t cuda_threads_per_block = 512;
    };

    /**
     * Training Results for a Currency Pair
     */
    struct PairTrainingResult
    {
        std::string pair_symbol;
        bool training_successful = false;

        // Performance metrics
        double overall_accuracy = 0.0;
        double high_confidence_accuracy = 0.0;
        double signal_rate = 0.0;
        double profitability_score = 0.0;  // (High-Conf Accuracy - 50) × Signal Rate

        // Optimized parameters
        sep::trading::QuantumTrainingConfig optimized_config;

        // Training metadata
        std::chrono::system_clock::time_point training_start;
        std::chrono::system_clock::time_point training_end;
        size_t training_samples_processed = 0;
        size_t convergence_iterations = 0;

        // Pattern analysis results
        std::vector<sep::quantum::Pattern> discovered_patterns;
        std::map<std::string, double> pattern_performance_scores;

        // Error information (if training failed)
        std::string error_message;
        std::string failure_reason;
    };

    /**
     * Professional Quantum Currency Pair Trainer
     * Implements production-grade quantum field harmonics training for forex pairs
     */
    class QuantumPairTrainer
    {
    public:
        explicit QuantumPairTrainer(const sep::trading::QuantumTrainingConfig& config = {});
        ~QuantumPairTrainer();

        // Core training interface
        std::future<sep::trading::PairTrainingResult> trainPairAsync(
            const std::string& pair_symbol);
        sep::trading::PairTrainingResult trainPair(const std::string& pair_symbol);

        // Batch training operations
        std::future<std::vector<sep::trading::PairTrainingResult>> trainMultiplePairsAsync(
            const std::vector<std::string>& pair_symbols);
        std::vector<sep::trading::PairTrainingResult> trainMultiplePairs(
            const std::vector<std::string>& pair_symbols);

        // Training management
        bool isTrainingActive() const { return training_active_.load(); }
        void cancelTraining();
        void pauseTraining();
        void resumeTraining();

        // Configuration management
        void updateConfig(const sep::trading::QuantumTrainingConfig& config);
        sep::trading::QuantumTrainingConfig getCurrentConfig() const;

        // Performance analysis
        std::vector<sep::trading::PairTrainingResult> getTrainingHistory() const;
        sep::trading::PairTrainingResult getLastTrainingResult(
            const std::string& pair_symbol) const;

        // Pattern management
        void saveTrainedPatterns(const std::string& pair_symbol,
                                 const std::string& file_path) const;
        bool loadTrainedPatterns(const std::string& pair_symbol, const std::string& file_path);

        // Validation and testing
        double validateTrainingResult(const std::string& pair_symbol,
                                      const std::vector<sep::connectors::MarketData>& test_data);

    private:
        // Core training implementation
        sep::trading::PairTrainingResult performQuantumTraining(const std::string& pair_symbol);

        // Data preparation
        std::vector<sep::connectors::MarketData> fetchTrainingData(const std::string& pair_symbol,
                                                                   size_t hours_back);
        std::vector<uint8_t> convertToBitstream(
            const std::vector<sep::connectors::MarketData>& market_data);

        // Quantum analysis
        sep::quantum::QFHResult performQFHAnalysis(const std::vector<uint8_t>& bitstream);
        std::vector<sep::quantum::Pattern> discoverPatterns(
            const std::vector<sep::connectors::MarketData>& market_data);

        // Parameter optimization
        sep::trading::QuantumTrainingConfig optimizeParameters(
            const std::string& pair_symbol,
            const std::vector<sep::connectors::MarketData>& training_data);
        double evaluateParameterSet(const sep::trading::QuantumTrainingConfig& config,
                                    const std::vector<sep::connectors::MarketData>& data);

        // Multi-timeframe analysis
        bool performTripleConfirmationAnalysis(
            const std::vector<sep::connectors::MarketData>& m1_data,
            const std::vector<sep::connectors::MarketData>& m5_data,
            const std::vector<sep::connectors::MarketData>& m15_data);

        // Performance calculation
        double calculateAccuracy(const std::vector<sep::connectors::MarketData>& data,
                                 const sep::trading::QuantumTrainingConfig& config);
        double calculateProfitabilityScore(double accuracy, double signal_rate);

        // Persistence
        void saveTrainingResult(const sep::trading::PairTrainingResult& result);
        sep::trading::PairTrainingResult loadTrainingResult(const std::string& pair_symbol);

        // Training state
        sep::trading::QuantumTrainingConfig config_;
        std::atomic<bool> training_active_{false};
        std::atomic<bool> training_paused_{false};
        std::atomic<bool> cancellation_requested_{false};

        // Threading and synchronization
        mutable std::mutex config_mutex_;
        mutable std::mutex results_mutex_;
        mutable std::mutex patterns_mutex_;

        // Component instances
        std::unique_ptr<sep::quantum::QFHBasedProcessor> qfh_processor_;
        std::unique_ptr<sep::quantum::manifold::QuantumManifoldOptimizer> manifold_optimizer_;
        std::unique_ptr<sep::quantum::PatternEvolutionBridge> pattern_evolver_;
        sep::engine::EngineFacade* engine_facade_;  // Singleton - use raw pointer
        std::unique_ptr<sep::connectors::OandaConnector> oanda_connector_;
        std::shared_ptr<sep::memory::IRedisManager> redis_manager_;

        // Training results storage
        std::map<std::string, std::vector<sep::trading::PairTrainingResult>> training_history_;
        std::map<std::string, std::vector<sep::quantum::Pattern>> trained_patterns_;

        // Training statistics
        std::atomic<size_t> total_training_sessions_{0};
        std::atomic<size_t> successful_training_sessions_{0};
        std::atomic<size_t> total_patterns_discovered_{0};

        // Cache management
        std::string getCacheKey(const std::string& pair_symbol) const;
        bool isCacheValid(const std::string& cache_key) const;
        void updateCache(const std::string& cache_key,
                         const sep::trading::PairTrainingResult& result);

        // Simple in-memory cache for training results (testing/demo purposes)
        mutable std::mutex cache_mutex_;
        std::map<std::string, sep::trading::PairTrainingResult> result_cache_;
    };

}  // namespace sep::trading
