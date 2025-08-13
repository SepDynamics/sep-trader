#pragma once

#include <queue>

#include "sep_precompiled.h"
#include "pair_manager.hpp"
#include "trading_state.hpp"
#include "weekly_cache_manager.hpp"

namespace sep::trading {

// Dynamic pair operation result
enum class PairOperationResult {
    SUCCESS,                // Operation completed successfully
    ALREADY_EXISTS,         // Pair already exists in system
    NOT_FOUND,              // Pair not found in system
    INVALID_SYMBOL,         // Invalid pair symbol format
    CACHE_NOT_READY,        // Cache validation failed
    TRADING_ACTIVE,         // Cannot modify while trading
    RESOURCE_LIMIT,         // System resource limits exceeded
    VALIDATION_FAILED,      // Pair validation checks failed
    NETWORK_ERROR,          // Network/connectivity issues
    CONFIGURATION_ERROR,    // Configuration problems
    SYSTEM_ERROR           // Internal system error
};

// Pair lifecycle stage
enum class PairLifecycleStage {
    INITIALIZING,    // Pair being added to system
    VALIDATING,      // Validating pair symbol and data
    CACHE_BUILDING,  // Building initial cache
    READY,           // Ready for trading
    TRADING,         // Currently active in trading
    PAUSING,         // Being paused/disabled
    REMOVING,        // Being removed from system
    ERROR            // Error state requiring intervention
};

// Runtime pair configuration
struct DynamicPairConfig {
    std::string symbol;
    std::string display_name;
    bool enabled;
    int priority;
    std::string category;
    std::string base_currency;
    std::string quote_currency;
    int pip_precision;
    double margin_rate;
    size_t max_position_size;
    std::string training_data_source;
    std::string model_template;
    std::unordered_map<std::string, std::string> custom_parameters;
    
    DynamicPairConfig() : enabled(false), priority(0), pip_precision(4), 
                         margin_rate(0.02), max_position_size(1000000) {}
};

// Pair operation details
struct PairOperation {
    std::string operation_id;
    std::string pair_symbol;
    std::string operation_type; // "ADD", "REMOVE", "ENABLE", "DISABLE", "UPDATE"
    ::sep::trading::PairLifecycleStage stage;
    ::sep::trading::PairOperationResult result;
    double progress_percentage;
    std::string status_message;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::chrono::duration<double> duration;
    std::vector<std::string> steps_completed;
    std::vector<std::string> errors;
    
    PairOperation() : stage(::sep::trading::PairLifecycleStage::INITIALIZING), 
                     result(::sep::trading::PairOperationResult::SUCCESS),
                     progress_percentage(0.0),
                     start_time(std::chrono::system_clock::now()),
                     duration(std::chrono::duration<double>::zero()) {}
};

// Resource allocation for pairs
struct ResourceAllocation {
    size_t max_total_pairs;
    size_t max_concurrent_operations;
    size_t max_trading_pairs;
    size_t max_cache_size_mb;
    size_t max_memory_per_pair_mb;
    double cpu_limit_percentage;
    size_t network_bandwidth_limit;
    
    ResourceAllocation() : max_total_pairs(50), max_concurrent_operations(5),
                          max_trading_pairs(20), max_cache_size_mb(10240),
                          max_memory_per_pair_mb(256), cpu_limit_percentage(80.0),
                          network_bandwidth_limit(1048576) {} // 1MB/s
};

// Dynamic pair management callbacks
// Temporarily comment out function types to test build
using PairLifecycleCallback =
    std::function<void(const std::string& pair_symbol, ::sep::trading::PairLifecycleStage old_stage,
                       ::sep::trading::PairLifecycleStage new_stage)>;
using PairOperationCallback = std::function<void(const ::sep::trading::PairOperation& operation)>;
using ResourceAllocationCallback =
    std::function<bool(const ::sep::trading::ResourceAllocation& current,
                       const ::sep::trading::ResourceAllocation& requested)>;

class DynamicPairManager {
public:
    DynamicPairManager();
    ~DynamicPairManager();

    // Dynamic pair operations
    std::string addPairAsync(const ::sep::trading::DynamicPairConfig& config);
    std::string removePairAsync(const std::string& pair_symbol, bool force = false);
    std::string enablePairAsync(const std::string& pair_symbol);
    std::string disablePairAsync(const std::string& pair_symbol);
    std::string updatePairConfigAsync(const std::string& pair_symbol, const ::sep::trading::DynamicPairConfig& config);
    
    // Synchronous operations (blocking)
    ::sep::trading::PairOperationResult addPair(const ::sep::trading::DynamicPairConfig& config, std::chrono::seconds timeout = std::chrono::seconds(300));
    ::sep::trading::PairOperationResult removePair(const std::string& pair_symbol, bool force = false, std::chrono::seconds timeout = std::chrono::seconds(60));
    ::sep::trading::PairOperationResult enablePair(const std::string& pair_symbol, std::chrono::seconds timeout = std::chrono::seconds(120));
    ::sep::trading::PairOperationResult disablePair(const std::string& pair_symbol, std::chrono::seconds timeout = std::chrono::seconds(30));
    
    // Operation monitoring
    ::sep::trading::PairOperation getOperation(const std::string& operation_id) const;
    std::vector<::sep::trading::PairOperation> getActiveOperations() const;
    std::vector<::sep::trading::PairOperation> getOperationHistory(const std::string& pair_symbol = "") const;
    bool isOperationComplete(const std::string& operation_id) const;
    bool cancelOperation(const std::string& operation_id);
    
    // Pair lifecycle management
    ::sep::trading::PairLifecycleStage getPairLifecycleStage(const std::string& pair_symbol) const;
    std::vector<std::string> getPairsByStage(::sep::trading::PairLifecycleStage stage) const;
    bool advancePairStage(const std::string& pair_symbol);
    bool rollbackPairStage(const std::string& pair_symbol);
    
    // Configuration management
    bool setDynamicPairConfig(const std::string& pair_symbol, const ::sep::trading::DynamicPairConfig& config);
    ::sep::trading::DynamicPairConfig getDynamicPairConfig(const std::string& pair_symbol) const;
    std::vector<::sep::trading::DynamicPairConfig> getAllDynamicConfigs() const;
    bool validatePairConfig(const ::sep::trading::DynamicPairConfig& config) const;
    
    // Resource management
    void setResourceAllocation(const ::sep::trading::ResourceAllocation& allocation);
    ::sep::trading::ResourceAllocation getResourceAllocation() const;
    ::sep::trading::ResourceAllocation getCurrentResourceUsage() const;
    bool hasAvailableResources(const ::sep::trading::DynamicPairConfig& config) const;
    void optimizeResourceAllocation();
    
    // Stream management integration
    bool addStreamForPair(const std::string& pair_symbol);
    bool removeStreamForPair(const std::string& pair_symbol);
    bool isStreamActive(const std::string& pair_symbol) const;
    std::vector<std::string> getActiveStreams() const;
    
    // Cache integration
    bool ensureCacheForPair(const std::string& pair_symbol);
    bool validateCacheForPair(const std::string& pair_symbol) const;
    bool warmupCacheForPair(const std::string& pair_symbol);
    void preemptiveCacheRefresh();
    
    // Trading integration
    bool isPairTradingReady(const std::string& pair_symbol) const;
    std::vector<std::string> getTradingReadyPairs() const;
    bool pausePairTrading(const std::string& pair_symbol);
    bool resumePairTrading(const std::string& pair_symbol);
    
    // Event system
    size_t addLifecycleCallback(::sep::trading::PairLifecycleCallback callback);
    void removeLifecycleCallback(size_t callback_id);
    size_t addOperationCallback(::sep::trading::PairOperationCallback callback);
    void removeOperationCallback(size_t callback_id);
    size_t addResourceCallback(::sep::trading::ResourceAllocationCallback callback);
    void removeResourceCallback(size_t callback_id);
    
    // Batch operations
    std::vector<std::string> addMultiplePairsAsync(const std::vector<::sep::trading::DynamicPairConfig>& configs);
    std::vector<std::string> removeMultiplePairsAsync(const std::vector<std::string>& pair_symbols, bool force = false);
    std::vector<::sep::trading::PairOperationResult> enableMultiplePairs(const std::vector<std::string>& pair_symbols);
    std::vector<::sep::trading::PairOperationResult> disableMultiplePairs(const std::vector<std::string>& pair_symbols);
    
    // System integration
    void integratePairManager(std::shared_ptr<sep::core::PairManager> pair_manager);
    void integrateTradingState(std::shared_ptr<sep::core::TradingState> trading_state);
    void integrateCacheManager(std::shared_ptr<sep::cache::WeeklyCacheManager> cache_manager);
    bool hasRequiredIntegrations() const;
    
    // Performance and monitoring
    size_t getTotalOperationsPerformed() const;
    size_t getSuccessfulOperations() const;
    size_t getFailedOperations() const;
    double getAverageOperationTime() const;
    std::chrono::duration<double> getSystemUptime() const;
    
    // Emergency controls
    void pauseAllOperations();
    void resumeAllOperations();
    bool isOperationsPaused() const;
    void emergencyStopAllPairs();
    void forceCleanupResources();
    
    // Additional helper methods for CLI interface
    std::vector<std::string> getAllPairs() const;
    bool isPairEnabled(const std::string& pair) const;

private:
    mutable std::mutex manager_mutex_;
    ::sep::trading::ResourceAllocation resource_allocation_;
    std::atomic<bool> operations_paused_{false};
    
    // Operation tracking
    std::unordered_map<std::string, ::sep::trading::PairOperation> active_operations_;
    std::vector<::sep::trading::PairOperation> operation_history_;
    std::atomic<size_t> operation_counter_{0};
    mutable std::mutex operations_mutex_;
    
    // Pair configurations
    std::unordered_map<std::string, ::sep::trading::DynamicPairConfig> dynamic_configs_;
    std::unordered_map<std::string, ::sep::trading::PairLifecycleStage> pair_stages_;
    mutable std::mutex config_mutex_;
    
    // Asynchronous operation processing
    std::unique_ptr<std::thread> operation_processor_thread_;
    std::atomic<bool> stop_processing_{false};
    std::condition_variable operation_cv_;
    std::queue<std::function<void()>> operation_queue_;
    mutable std::mutex queue_mutex_;
    
    // External integrations
    std::shared_ptr<sep::core::PairManager> pair_manager_;
    std::shared_ptr<sep::core::TradingState> trading_state_;
    std::shared_ptr<sep::cache::WeeklyCacheManager> cache_manager_;
    
    // Event callbacks
    std::vector<::sep::trading::PairLifecycleCallback> lifecycle_callbacks_;
    std::vector<::sep::trading::PairOperationCallback> operation_callbacks_;
    std::vector<::sep::trading::ResourceAllocationCallback> resource_callbacks_;
    mutable std::mutex callbacks_mutex_;
    
    // Performance tracking
    std::atomic<size_t> total_operations_{0};
    std::atomic<size_t> successful_operations_{0};
    std::atomic<size_t> failed_operations_{0};
    std::atomic<std::chrono::duration<double>> total_operation_time_{std::chrono::duration<double>::zero()};
    std::chrono::system_clock::time_point system_start_time_;
    
    // Internal operation methods
    void operationProcessorLoop();
    ::sep::trading::PairOperationResult performAddPair(const ::sep::trading::DynamicPairConfig& config, ::sep::trading::PairOperation& operation);
    ::sep::trading::PairOperationResult performRemovePair(const std::string& pair_symbol, bool force, ::sep::trading::PairOperation& operation);
    ::sep::trading::PairOperationResult performEnablePair(const std::string& pair_symbol, ::sep::trading::PairOperation& operation);
    ::sep::trading::PairOperationResult performDisablePair(const std::string& pair_symbol, ::sep::trading::PairOperation& operation);
    
    // Lifecycle management
    bool transitionPairStage(const std::string& pair_symbol, ::sep::trading::PairLifecycleStage new_stage);
    bool validateStageTransition(::sep::trading::PairLifecycleStage current, ::sep::trading::PairLifecycleStage target) const;
    std::vector<std::string> getStageTransitionSteps(::sep::trading::PairLifecycleStage current, ::sep::trading::PairLifecycleStage target) const;
    
    // Validation methods
    bool validatePairSymbolFormat(const std::string& symbol) const;
    bool validatePairAvailability(const std::string& symbol) const;
    bool validateResourceRequirements(const ::sep::trading::DynamicPairConfig& config) const;
    bool validateMarketData(const std::string& symbol) const;
    
    // Resource management
    ::sep::trading::ResourceAllocation calculateCurrentUsage() const;
    bool allocateResourcesForPair(const std::string& pair_symbol, const ::sep::trading::DynamicPairConfig& config);
    void deallocateResourcesForPair(const std::string& pair_symbol);
    bool checkResourceLimits(const ::sep::trading::ResourceAllocation& requested) const;
    
    // Stream management
    bool initializeStreamForPair(const std::string& pair_symbol);
    bool cleanupStreamForPair(const std::string& pair_symbol);
    bool validateStreamConnection(const std::string& pair_symbol) const;
    
    // Event notification
    void notifyLifecycleChange(const std::string& pair_symbol, ::sep::trading::PairLifecycleStage old_stage, ::sep::trading::PairLifecycleStage new_stage);
    void notifyOperationUpdate(const ::sep::trading::PairOperation& operation);
    bool requestResourceAllocation(const ::sep::trading::ResourceAllocation& requested);
    
    // Utility methods
    std::string generateOperationId();
    ::sep::trading::PairOperation createOperation(const std::string& pair_symbol, const std::string& operation_type);
    void updateOperationProgress(::sep::trading::PairOperation& operation, double progress, const std::string& message);
    void completeOperation(::sep::trading::PairOperation& operation, ::sep::trading::PairOperationResult result);
    
    // Cleanup and maintenance
    void performPeriodicCleanup();
    void cleanupCompletedOperations();
    void validateSystemConsistency();
};

// Utility functions
std::string pairOperationResultToString(::sep::trading::PairOperationResult result);
::sep::trading::PairOperationResult stringToPairOperationResult(const std::string& result_str);
std::string pairLifecycleStageToString(::sep::trading::PairLifecycleStage stage);
::sep::trading::PairLifecycleStage stringToPairLifecycleStage(const std::string& stage_str);
bool isPairOperationResultSuccess(::sep::trading::PairOperationResult result);
bool isPairLifecycleStageActive(::sep::trading::PairLifecycleStage stage);

// Global dynamic pair manager instance
DynamicPairManager& getGlobalDynamicPairManager();

} // namespace sep::trading