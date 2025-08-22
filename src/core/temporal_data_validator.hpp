#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include "core/quantum_pair_trainer.hpp"
#include "io/oanda_connector.h"

namespace sep::validation
{
    /**
     * Temporal Data Validation Configuration
     * Advanced configuration for rigorous temporal analysis
     */
    struct TemporalValidationConfig
    {
        // Walk-forward validation parameters
        size_t walk_forward_window_hours = 168;     // 1 week sliding window
        size_t walk_forward_step_hours = 24;        // Daily advancement
        size_t minimum_training_hours = 720;        // 30 days minimum training
        size_t maximum_training_hours = 8760;       // 1 year maximum training
        
        // 3D Volumetric Analysis Parameters
        size_t volume_depth_levels = 10;            // Price depth levels
        size_t volume_time_buckets = 24;            // Hourly volume buckets
        size_t volume_pattern_buckets = 50;         // Pattern complexity buckets
        
        // Temporal Partitioning Configuration (in hours)
        std::vector<size_t> temporal_partition_hours = {
            1,    // M1-based hourly
            4,    // 4-hour sessions
            24,   // Daily
            168   // Weekly
        };
        
        // Multi-timeframe coherence requirements
        double m1_m5_coherence_threshold = 0.75;
        double m5_m15_coherence_threshold = 0.70;
        double m15_h1_coherence_threshold = 0.65;
        double cross_timeframe_weight = 0.85;
        
        // CUDA Acceleration Settings
        bool enable_cuda_acceleration = true;
        size_t cuda_batch_size = 512;
        size_t cuda_thread_blocks = 256;
        size_t cuda_shared_memory_kb = 48;
        
        // Fidelity Assessment Thresholds
        double minimum_accuracy_threshold = 0.55;    // 55% minimum
        double high_confidence_threshold = 0.65;     // 65% high confidence
        double exceptional_accuracy_threshold = 0.75; // 75% exceptional
        
        // Performance Benchmarking
        size_t benchmark_iterations = 1000;
        double benchmark_tolerance = 1e-6;
        bool enable_performance_profiling = true;
    };

    /**
     * Temporal Validation Results
     * Comprehensive results from temporal analysis
     */
    struct TemporalValidationResult
    {
        // Walk-forward validation results
        std::vector<double> walk_forward_accuracies;
        double average_walk_forward_accuracy = 0.0;
        double walk_forward_stability = 0.0;
        double walk_forward_consistency = 0.0;
        
        // 3D Volumetric Analysis Results
        std::map<size_t, std::vector<double>> volume_depth_analysis;
        std::map<size_t, double> volume_time_correlations;
        std::map<size_t, double> volume_pattern_strengths;
        double volumetric_coherence_score = 0.0;
        
        // Temporal Partition Results
        std::map<size_t, double> partition_accuracies;      // hours -> accuracy
        std::map<size_t, double> partition_stabilities;     // hours -> stability
        double temporal_consistency_score = 0.0;
        
        // Multi-timeframe Coherence Results
        double m1_m5_coherence = 0.0;
        double m5_m15_coherence = 0.0;
        double m15_h1_coherence = 0.0;
        double overall_timeframe_coherence = 0.0;
        
        // Fidelity Assessment
        double overall_fidelity_score = 0.0;
        bool passes_minimum_threshold = false;
        bool achieves_high_confidence = false;
        bool achieves_exceptional_accuracy = false;
        
        // Performance Metrics
        std::chrono::nanoseconds cuda_processing_time{0};
        std::chrono::nanoseconds cpu_processing_time{0};
        double cuda_speedup_factor = 0.0;
        size_t memory_usage_bytes = 0;
        
        // Validation Metadata
        std::string pair_symbol;
        std::chrono::system_clock::time_point validation_timestamp;
        size_t total_samples_analyzed = 0;
        size_t valid_samples_processed = 0;
        std::string validation_status;
        std::vector<std::string> warning_messages;
        std::vector<std::string> error_messages;
    };

    /**
     * Advanced Temporal Data Validator
     * Comprehensive validation system for quantum trading models
     */
    class TemporalDataValidator
    {
    public:
        explicit TemporalDataValidator(const TemporalValidationConfig& config = {});
        ~TemporalDataValidator();

        // Primary validation interface
        std::future<TemporalValidationResult> validatePairAsync(
            const std::string& pair_symbol);
        TemporalValidationResult validatePair(const std::string& pair_symbol);

        // Walk-forward validation
        std::vector<double> performWalkForwardValidation(
            const std::string& pair_symbol,
            const std::vector<sep::connectors::MarketData>& historical_data);
        
        // 3D Volumetric analysis
        TemporalValidationResult perform3DVolumetricAnalysis(
            const std::string& pair_symbol,
            const std::vector<sep::connectors::MarketData>& market_data);
        
        // Temporal partitioning
        std::map<size_t, double> performTemporalPartitioning(
            const std::vector<sep::connectors::MarketData>& market_data);
        
        // Multi-timeframe coherence analysis
        double calculateMultiTimeframeCoherence(
            const std::vector<sep::connectors::MarketData>& m1_data,
            const std::vector<sep::connectors::MarketData>& m5_data,
            const std::vector<sep::connectors::MarketData>& m15_data,
            const std::vector<sep::connectors::MarketData>& h1_data);
        
        // CUDA-accelerated forward window volume sampling
        std::vector<double> performCudaForwardWindowSampling(
            const std::vector<sep::connectors::MarketData>& market_data,
            size_t forward_window_size);
        
        // Fidelity assessment and benchmarking
        double calculateFidelityScore(const TemporalValidationResult& results);
        std::map<std::string, double> performComprehensiveBenchmarking(
            const std::string& pair_symbol);
        
        // Configuration management
        void updateConfig(const TemporalValidationConfig& config);
        TemporalValidationConfig getCurrentConfig() const;
        
        // Performance profiling
        void enablePerformanceProfiling(bool enable);
        std::map<std::string, std::chrono::nanoseconds> getProfilingResults() const;
        
    private:
        // Core validation implementations
        TemporalValidationResult executeValidation(const std::string& pair_symbol);
        
        // Walk-forward implementation
        double performSingleWalkForward(
            const std::vector<sep::connectors::MarketData>& training_data,
            const std::vector<sep::connectors::MarketData>& validation_data);
        
        // 3D volumetric calculations
        std::vector<double> calculateVolumeDepthLevels(
            const std::vector<sep::connectors::MarketData>& market_data,
            size_t depth_levels);
        std::map<size_t, double> calculateVolumeTimeCorrelations(
            const std::vector<sep::connectors::MarketData>& market_data);
        std::map<size_t, double> analyzeVolumePatternStrengths(
            const std::vector<sep::connectors::MarketData>& market_data);
        
        // CUDA kernel implementations
        void initializeCudaContext();
        void cleanupCudaContext();
        std::vector<float> launchCudaVolumetricKernel(
            const std::vector<float>& input_data,
            size_t batch_size);
        
        // Temporal coherence calculations
        double calculateTimeframeCoherence(
            const std::vector<sep::connectors::MarketData>& data1,
            const std::vector<sep::connectors::MarketData>& data2);
        
        // Performance monitoring
        void startProfiling(const std::string& operation_name);
        void endProfiling(const std::string& operation_name);
        
        // Data preparation utilities
        std::vector<sep::connectors::MarketData> fetchMultiTimeframeData(
            const std::string& pair_symbol,
            size_t timeframe_hours,
            size_t hours_back);
        std::vector<float> convertToFloatVector(
            const std::vector<sep::connectors::MarketData>& market_data);
        
        // Configuration and state
        TemporalValidationConfig config_;
        std::unique_ptr<sep::trading::QuantumPairTrainer> quantum_trainer_;
        std::unique_ptr<sep::connectors::OandaConnector> oanda_connector_;
        
        // CUDA resources
        void* cuda_context_ = nullptr;
        void* cuda_stream_ = nullptr;
        float* cuda_device_memory_ = nullptr;
        size_t cuda_memory_size_ = 0;
        
        // Performance profiling
        mutable std::mutex profiling_mutex_;
        std::map<std::string, std::chrono::high_resolution_clock::time_point> profiling_start_times_;
        std::map<std::string, std::chrono::nanoseconds> profiling_results_;
        bool profiling_enabled_ = false;
        
        // Threading and synchronization
        mutable std::mutex config_mutex_;
        mutable std::mutex validation_mutex_;
        std::atomic<bool> cuda_initialized_{false};
    };

    // Factory functions for easy instantiation
    std::unique_ptr<TemporalDataValidator> createTemporalValidator(
        const TemporalValidationConfig& config = {});
    
    // Utility functions for validation analysis
    double calculateValidationStability(const std::vector<double>& accuracies);
    double calculateValidationConsistency(const std::vector<double>& accuracies);
    std::string formatValidationReport(const TemporalValidationResult& result);

} // namespace sep::validation