#pragma once

// Use simple C types to avoid type pollution from precompiled headers
#include <cstdint>
#include <cstring>  // For strcpy and memset

namespace sep::training
{
    /**
     * Training Orchestration Configuration (C-Style)
     * Simplified configuration structure using basic types
     */
    struct TrainingOrchestrationConfig
    {
        // Training Pipeline Configuration
        bool enable_multi_pair_training = true;
        bool enable_incremental_learning = true;
        bool enable_ensemble_training = false;
        uint32_t max_concurrent_trainings = 4;
        
        // Data Management
        uint32_t historical_data_days = 365;
        uint32_t validation_split_percentage = 20;
        bool enable_data_augmentation = true;
        uint32_t data_refresh_interval_hours = 6;
        
        // Model Selection and Ensemble
        bool enable_model_selection = true;
        uint32_t ensemble_size = 3;
        double ensemble_voting_threshold = 0.6;
        bool enable_model_versioning = true;
        
        // Training Quality Control
        double min_accuracy_threshold = 0.55;
        double target_accuracy = 0.65;
        uint32_t max_training_iterations = 1000;
        double early_stopping_patience = 0.01;
        
        // Performance and Resource Management
        bool enable_gpu_acceleration = true;
        uint32_t gpu_memory_limit_mb = 4096;
        uint32_t cpu_thread_limit = 8;
        bool enable_distributed_training = false;
        
        // Monitoring and Logging
        bool enable_progress_callbacks = true;
        bool enable_comprehensive_logging = true;
        char output_directory[256]; // Fixed-size C string
        bool enable_tensorboard_logging = false;
        
        // Constructor to initialize C string
        TrainingOrchestrationConfig() {
            strcpy(output_directory, "output/training/");
        }
    };

    /**
     * Training Session Status (C-Style)
     * Current state of training operations
     */
    struct TrainingSessionStatus
    {
        // Session Identification (Fixed-size C strings)
        char session_id[64];
        char pair_symbol[16];
        char training_mode[32]; // "initial", "incremental", "ensemble"
        
        // Progress Tracking
        uint32_t current_iteration = 0;
        uint32_t total_iterations = 0;
        double progress_percentage = 0.0;
        double current_accuracy = 0.0;
        double best_accuracy = 0.0;
        
        // Resource Utilization
        double cpu_usage_percentage = 0.0;
        double gpu_usage_percentage = 0.0;
        uint32_t memory_usage_mb = 0;
        double training_speed_samples_per_second = 0.0;
        
        // Training Metrics
        double current_loss = 0.0;
        double validation_loss = 0.0;
        double learning_rate = 0.0;
        uint32_t samples_processed = 0;
        
        // Status Information (Fixed-size C strings)
        char status[32]; // "initializing", "training", "validating", "completed", "failed"
        char current_phase[64]; // "data_preparation", "model_training", "validation", "optimization"
        uint64_t start_time_epoch = 0;
        uint64_t estimated_completion_time_epoch = 0;
        
        // Error and Warning Information (Fixed arrays)
        char warnings[10][256]; // Up to 10 warnings, 256 chars each
        char errors[10][256];   // Up to 10 errors, 256 chars each
        uint32_t warning_count = 0;
        uint32_t error_count = 0;
        bool has_critical_errors = false;
        
        // Constructor to initialize C strings
        TrainingSessionStatus() {
            memset(session_id, 0, sizeof(session_id));
            memset(pair_symbol, 0, sizeof(pair_symbol));
            memset(training_mode, 0, sizeof(training_mode));
            memset(status, 0, sizeof(status));
            memset(current_phase, 0, sizeof(current_phase));
            memset(warnings, 0, sizeof(warnings));
            memset(errors, 0, sizeof(errors));
        }
    };

    /**
     * Simple Temporal Validation Result (C-Style)
     */
    struct TemporalValidationResult
    {
        // Basic validation metrics
        double overall_fidelity_score = 0.0;
        bool passes_minimum_threshold = false;
        bool achieves_high_confidence = false;
        double temporal_consistency_score = 0.0;
        double overall_timeframe_coherence = 0.0;
        char validation_status[64];
        
        // Constructor
        TemporalValidationResult() {
            memset(validation_status, 0, sizeof(validation_status));
        }
    };

    /**
     * Training Results Summary (C-Style)
     * Comprehensive results from training operations
     */
    struct TrainingResultsSummary
    {
        // Training Session Identification
        char session_id[64];
        char pair_symbol[16];
        uint64_t completion_timestamp_epoch = 0;
        double total_training_duration_seconds = 0.0;
        
        // Accuracy and Performance Metrics
        double final_training_accuracy = 0.0;
        double final_validation_accuracy = 0.0;
        double peak_accuracy_achieved = 0.0;
        double average_training_loss = 0.0;
        double final_validation_loss = 0.0;
        
        // Training Convergence Analysis
        uint32_t total_iterations_completed = 0;
        bool converged_successfully = false;
        bool early_stopping_triggered = false;
        char stopping_reason[128]; // "convergence", "max_iterations", "accuracy_target", "manual", "error"
        
        // Resource Utilization Summary
        double average_cpu_usage = 0.0;
        double peak_gpu_usage = 0.0;
        uint32_t peak_memory_usage_mb = 0;
        double average_training_speed = 0.0;
        
        // Model Quality Assessment
        double confidence_score = 0.0;
        double stability_index = 0.0;
        double robustness_metric = 0.0;
        uint32_t samples_processed_total = 0;
        
        // Model Comparison Results
        char best_model_variant[64];
        double model_variant_accuracies[5]; // Support up to 5 variants
        uint32_t variant_count = 0;
        bool ensemble_outperformed_single = false;
        
        // Validation Results
        TemporalValidationResult temporal_validation;
        bool passed_accuracy_requirements = false;
        char performance_grade[4]; // A, B, C, D, F
        char improvement_recommendations[5][256]; // Up to 5 recommendations
        uint32_t recommendation_count = 0;
        
        // Constructor
        TrainingResultsSummary() {
            memset(session_id, 0, sizeof(session_id));
            memset(pair_symbol, 0, sizeof(pair_symbol));
            memset(stopping_reason, 0, sizeof(stopping_reason));
            memset(best_model_variant, 0, sizeof(best_model_variant));
            memset(performance_grade, 0, sizeof(performance_grade));
            memset(improvement_recommendations, 0, sizeof(improvement_recommendations));
        }
    };

    /**
     * Quantum Training Coordinator (C-Style Interface)
     * Professional-grade training orchestration system
     */
    class QuantumTrainingCoordinator
    {
    public:
        explicit QuantumTrainingCoordinator(const TrainingOrchestrationConfig& config = {});
        ~QuantumTrainingCoordinator();
        
        // Primary Training Orchestration Interface (C-Style)
        bool startTrainingSession(const char* pair_symbol, char* session_id_out);
        bool startMultiPairTraining(const char** pair_symbols, uint32_t pair_count);
        
        // Advanced Training Modes
        bool startEnsembleTraining(const char* pair_symbol, char* session_id_out);
        bool startIncrementalTraining(const char* pair_symbol, const char* base_model_path, char* session_id_out);
        
        // Training Control and Management
        bool pauseTraining(const char* session_id);
        bool resumeTraining(const char* session_id);
        bool stopTraining(const char* session_id);
        void stopAllTraining();
        
        // Session Monitoring and Status
        bool getSessionStatus(const char* session_id, TrainingSessionStatus* status_out) const;
        uint32_t getAllActiveSessionsStatus(TrainingSessionStatus* statuses_out, uint32_t max_sessions) const;
        bool isTrainingInProgress() const;
        uint32_t getActiveSessionCount() const;
        
        // Model Management (C-Style)
        bool deployBestModel(const char* pair_symbol, char* deployed_model_path_out);
        bool rollbackModel(const char* pair_symbol, const char* version);
        uint32_t getAvailableModelVersions(const char* pair_symbol, char versions_out[][64], uint32_t max_versions) const;
        bool compareModelVersions(const char* pair_symbol, const char** versions, uint32_t version_count, TrainingResultsSummary* comparison_out) const;
        
        // Performance Analysis and Optimization
        bool performAutomaticHyperparameterTuning(const char* pair_symbol);
        bool runComprehensiveValidation(const char* pair_symbol, TrainingResultsSummary* results_out);
        void optimizeResourceAllocation();
        uint32_t analyzeTrainingBottlenecks(char bottleneck_names_out[][64], double bottleneck_scores_out[], uint32_t max_bottlenecks) const;
        
        // Configuration and Settings Management
        void updateConfig(const TrainingOrchestrationConfig& config);
        TrainingOrchestrationConfig getCurrentConfig() const;
        bool saveConfiguration(const char* config_file_path) const;
        bool loadConfiguration(const char* config_file_path);
        
        // Reporting and Export Functions (C-Style)
        bool generateTrainingReport(const char* session_id, char* report_content_out, uint32_t max_content_size) const;
        bool exportTrainingMetrics(const char* session_id, const char* output_path) const;
        bool generateComprehensiveSystemReport(const char* output_path) const;
        
        // Callback Support (C-Style function pointers)
        typedef void (*ProgressCallback)(const TrainingSessionStatus* status);
        typedef void (*CompletionCallback)(const TrainingResultsSummary* results);
        typedef void (*ErrorCallback)(const char* session_id, const char* error_message);
        
        void setProgressCallback(ProgressCallback callback);
        void setCompletionCallback(CompletionCallback callback);
        void setErrorCallback(ErrorCallback callback);

    private:
        // Configuration
        TrainingOrchestrationConfig config_;
        
        // Internal state management (C-style arrays)
        TrainingSessionStatus active_sessions_[16]; // Support up to 16 concurrent sessions
        uint32_t active_session_count_;
        
        // Callback handlers
        ProgressCallback progress_callback_;
        CompletionCallback completion_callback_;
        ErrorCallback error_callback_;
        
        // Internal utility methods
        void generateSessionId(char* session_id_out) const;
        bool initializeTrainingEnvironment(const char* pair_symbol);
        void cleanupSession(const char* session_id);
        bool analyzeTrainingResults(const char* session_id, TrainingResultsSummary* results_out) const;
        void notifyProgressUpdate(const char* session_id);
        void handleTrainingError(const char* session_id, const char* error_message);
        int findSessionIndex(const char* session_id) const;
    };

} // namespace sep::training