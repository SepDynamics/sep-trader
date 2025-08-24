
#include "core/dynamic_pair_manager.hpp"
#include <algorithm>
#include <regex>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

namespace sep::trading {

DynamicPairManager::DynamicPairManager() 
    : system_start_time_(std::chrono::system_clock::now()) {
    
    // Initialize default resource allocation
    resource_allocation_ = ResourceAllocation();
    
    // Start operation processor thread
    stop_processing_.store(false);
    operation_processor_thread_ = std::make_unique<std::thread>(&DynamicPairManager::operationProcessorLoop, this);
}

DynamicPairManager::~DynamicPairManager() {
    // Stop operation processing
    stop_processing_.store(true);
    operation_cv_.notify_all();
    
    if (operation_processor_thread_ && operation_processor_thread_->joinable()) {
        operation_processor_thread_->join();
    }
}

std::string DynamicPairManager::addPairAsync(const DynamicPairConfig& config) {
    if (operations_paused_.load()) {
        return ""; // Operations are paused
    }
    
    std::string operation_id = generateOperationId();
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    operation_queue_.push([this, config, operation_id]() {
        std::lock_guard<std::mutex> ops_lock(operations_mutex_);
        
        PairOperation operation = createOperation(config.symbol, "ADD");
        operation.operation_id = operation_id;
        
        active_operations_[operation_id] = operation;
        
        PairOperationResult result = performAddPair(config, active_operations_[operation_id]);
        completeOperation(active_operations_[operation_id], result);
    });
    
    operation_cv_.notify_one();
    return operation_id;
}

std::string DynamicPairManager::removePairAsync(const std::string& pair_symbol, bool force) {
    if (operations_paused_.load()) {
        return ""; // Operations are paused
    }
    
    std::string operation_id = generateOperationId();
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    operation_queue_.push([this, pair_symbol, force, operation_id]() {
        std::lock_guard<std::mutex> ops_lock(operations_mutex_);
        
        PairOperation operation = createOperation(pair_symbol, "REMOVE");
        operation.operation_id = operation_id;
        
        active_operations_[operation_id] = operation;
        
        PairOperationResult result = performRemovePair(pair_symbol, force, active_operations_[operation_id]);
        completeOperation(active_operations_[operation_id], result);
    });
    
    operation_cv_.notify_one();
    return operation_id;
}

PairOperationResult DynamicPairManager::addPair(const DynamicPairConfig& config, std::chrono::seconds timeout) {
    std::string operation_id = addPairAsync(config);
    if (operation_id.empty()) {
        return PairOperationResult::SYSTEM_ERROR;
    }
    
    // Wait for completion
    auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start_time < timeout) {
        if (isOperationComplete(operation_id)) {
            return getOperation(operation_id).result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Timeout
    cancelOperation(operation_id);
    return PairOperationResult::SYSTEM_ERROR;
}

PairOperation DynamicPairManager::getOperation(const std::string& operation_id) const {
    std::lock_guard<std::mutex> lock(operations_mutex_);
    
    auto it = active_operations_.find(operation_id);
    if (it != active_operations_.end()) {
        return it->second;
    }
    
    // Check operation history
    for (const auto& op : operation_history_) {
        if (op.operation_id == operation_id) {
            return op;
        }
    }
    
    // Return empty operation if not found
    return PairOperation();
}

std::vector<PairOperation> DynamicPairManager::getActiveOperations() const {
    std::lock_guard<std::mutex> lock(operations_mutex_);
    
    std::vector<PairOperation> operations;
    operations.reserve(active_operations_.size());
    
    for (const auto& op_pair : active_operations_) {
        operations.push_back(op_pair.second);
    }
    
    return operations;
}

bool DynamicPairManager::isOperationComplete(const std::string& operation_id) const {
    std::lock_guard<std::mutex> lock(operations_mutex_);
    
    // Check if operation is still active
    auto it = active_operations_.find(operation_id);
    if (it != active_operations_.end()) {
        return it->second.end_time != std::chrono::system_clock::time_point{};
    }
    
    // Check operation history
    for (const auto& op : operation_history_) {
        if (op.operation_id == operation_id) {
            return true; // Found in history means it's complete
        }
    }
    
    return false; // Operation not found
}

bool DynamicPairManager::cancelOperation(const std::string& operation_id) {
    std::lock_guard<std::mutex> lock(operations_mutex_);
    
    auto it = active_operations_.find(operation_id);
    if (it != active_operations_.end()) {
        it->second.result = PairOperationResult::SYSTEM_ERROR;
        it->second.status_message = "Operation cancelled";
        completeOperation(it->second, PairOperationResult::SYSTEM_ERROR);
        return true;
    }
    
    return false;
}

PairLifecycleStage DynamicPairManager::getPairLifecycleStage(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    auto it = pair_stages_.find(pair_symbol);
    if (it != pair_stages_.end()) {
        return it->second;
    }
    
    return PairLifecycleStage::ERROR;
}

bool DynamicPairManager::setDynamicPairConfig(const std::string& pair_symbol, const DynamicPairConfig& config) {
    if (!validatePairConfig(config)) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(config_mutex_);
    dynamic_configs_[pair_symbol] = config;
    
    return true;
}

DynamicPairConfig DynamicPairManager::getDynamicPairConfig(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    auto it = dynamic_configs_.find(pair_symbol);
    if (it != dynamic_configs_.end()) {
        return it->second;
    }
    
    return DynamicPairConfig(); // Return default config
}

bool DynamicPairManager::validatePairConfig(const DynamicPairConfig& config) const {
    return validatePairSymbolFormat(config.symbol) &&
           !config.display_name.empty() &&
           config.pip_precision > 0 &&
           config.margin_rate > 0.0 &&
           config.max_position_size > 0;
}

void DynamicPairManager::setResourceAllocation(const ResourceAllocation& allocation) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    // Notify callbacks about resource allocation change
    bool approved = true;
    {
        std::lock_guard<std::mutex> cb_lock(callbacks_mutex_);
        for (const auto& callback : resource_callbacks_) {
            try {
                if (!callback(resource_allocation_, allocation)) {
                    approved = false;
                    break;
                }
            } catch (const std::exception&) {
                // Ignore callback errors
            }
        }
    }
    
    if (approved) {
        resource_allocation_ = allocation;
    }
}

ResourceAllocation DynamicPairManager::getResourceAllocation() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return resource_allocation_;
}

ResourceAllocation DynamicPairManager::getCurrentResourceUsage() const {
    return calculateCurrentUsage();
}

bool DynamicPairManager::hasAvailableResources(const DynamicPairConfig& config) const {
    return validateResourceRequirements(config);
}

bool DynamicPairManager::isPairTradingReady(const std::string& pair_symbol) const {
    PairLifecycleStage stage = getPairLifecycleStage(pair_symbol);
    return stage == PairLifecycleStage::READY || stage == PairLifecycleStage::TRADING;
}

size_t DynamicPairManager::getTotalOperationsPerformed() const {
    return total_operations_.load();
}

size_t DynamicPairManager::getSuccessfulOperations() const {
    return successful_operations_.load();
}

size_t DynamicPairManager::getFailedOperations() const {
    return failed_operations_.load();
}

double DynamicPairManager::getAverageOperationTime() const {
    size_t total_ops = total_operations_.load();
    if (total_ops == 0) {
        return 0.0;
    }
    
    auto total_time = total_operation_time_.load();
    return total_time.count() / total_ops;
}

std::chrono::duration<double> DynamicPairManager::getSystemUptime() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now - system_start_time_);
}

void DynamicPairManager::pauseAllOperations() {
    operations_paused_.store(true);
}

void DynamicPairManager::resumeAllOperations() {
    operations_paused_.store(false);
    operation_cv_.notify_all();
}

bool DynamicPairManager::isOperationsPaused() const {
    return operations_paused_.load();
}

void DynamicPairManager::operationProcessorLoop() {
    while (!stop_processing_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for operations or stop signal
        operation_cv_.wait(lock, [this] { 
            return !operation_queue_.empty() || stop_processing_.load(); 
        });
        
        while (!operation_queue_.empty() && !operations_paused_.load()) {
            auto operation = operation_queue_.front();
            operation_queue_.pop();
            
            lock.unlock();
            
            try {
                operation();
            } catch (const std::exception& e) {
                // Log error
            }
            
            lock.lock();
        }
    }
}

PairOperationResult DynamicPairManager::performAddPair(const DynamicPairConfig& config, PairOperation& operation) {
    updateOperationProgress(operation, 10.0, "Validating pair configuration");
    
    if (!validatePairConfig(config)) {
        return PairOperationResult::VALIDATION_FAILED;
    }
    
    updateOperationProgress(operation, 20.0, "Checking resource availability");
    
    if (!validateResourceRequirements(config)) {
        return PairOperationResult::RESOURCE_LIMIT;
    }
    
    updateOperationProgress(operation, 40.0, "Initializing pair in system");
    
    // Set lifecycle stage
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        pair_stages_[config.symbol] = PairLifecycleStage::INITIALIZING;
        dynamic_configs_[config.symbol] = config;
    }
    
    updateOperationProgress(operation, 60.0, "Setting up data streams");
    
    // Simulate stream setup
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    updateOperationProgress(operation, 80.0, "Validating cache requirements");
    
    // Simulate cache validation
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    updateOperationProgress(operation, 95.0, "Finalizing pair setup");
    
    // Transition to ready state
    transitionPairStage(config.symbol, PairLifecycleStage::READY);
    
    updateOperationProgress(operation, 100.0, "Pair added successfully");
    
    return PairOperationResult::SUCCESS;
}

PairOperationResult DynamicPairManager::performRemovePair(const std::string& pair_symbol, bool force, PairOperation& operation) {
    updateOperationProgress(operation, 10.0, "Checking pair status");
    
    PairLifecycleStage current_stage = getPairLifecycleStage(pair_symbol);
    
    if (current_stage == PairLifecycleStage::TRADING && !force) {
        return PairOperationResult::TRADING_ACTIVE;
    }
    
    updateOperationProgress(operation, 30.0, "Stopping trading activities");
    
    // Transition to removing stage
    transitionPairStage(pair_symbol, PairLifecycleStage::REMOVING);
    
    updateOperationProgress(operation, 50.0, "Cleaning up resources");
    
    // Simulate resource cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    updateOperationProgress(operation, 80.0, "Removing from system");
    
    // Remove from configurations
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        dynamic_configs_.erase(pair_symbol);
        pair_stages_.erase(pair_symbol);
    }
    
    updateOperationProgress(operation, 100.0, "Pair removed successfully");
    
    return PairOperationResult::SUCCESS;
}

PairOperationResult DynamicPairManager::performEnablePair(const std::string& pair_symbol, PairOperation& operation) {
    updateOperationProgress(operation, 20.0, "Validating pair readiness");
    
    PairLifecycleStage current_stage = getPairLifecycleStage(pair_symbol);
    
    if (current_stage != PairLifecycleStage::READY) {
        return PairOperationResult::VALIDATION_FAILED;
    }
    
    updateOperationProgress(operation, 60.0, "Activating trading");
    
    // Transition to trading stage
    transitionPairStage(pair_symbol, PairLifecycleStage::TRADING);
    
    updateOperationProgress(operation, 100.0, "Pair enabled for trading");
    
    return PairOperationResult::SUCCESS;
}

PairOperationResult DynamicPairManager::performDisablePair(const std::string& pair_symbol, PairOperation& operation) {
    updateOperationProgress(operation, 30.0, "Stopping trading activities");
    
    // Transition to pausing stage
    transitionPairStage(pair_symbol, PairLifecycleStage::PAUSING);
    
    updateOperationProgress(operation, 70.0, "Finalizing disable");
    
    // Transition to ready stage
    transitionPairStage(pair_symbol, PairLifecycleStage::READY);
    
    updateOperationProgress(operation, 100.0, "Pair disabled from trading");
    
    return PairOperationResult::SUCCESS;
}

bool DynamicPairManager::transitionPairStage(const std::string& pair_symbol, PairLifecycleStage new_stage) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    auto it = pair_stages_.find(pair_symbol);
    PairLifecycleStage old_stage = (it != pair_stages_.end()) ? it->second : PairLifecycleStage::ERROR;
    
    if (!validateStageTransition(old_stage, new_stage)) {
        return false;
    }
    
    pair_stages_[pair_symbol] = new_stage;
    
    // Notify lifecycle change
    notifyLifecycleChange(pair_symbol, old_stage, new_stage);
    
    return true;
}

bool DynamicPairManager::validateStageTransition(PairLifecycleStage current, PairLifecycleStage target) const {
    // Define valid transitions
    switch (current) {
        case PairLifecycleStage::INITIALIZING:
            return target == PairLifecycleStage::VALIDATING || target == PairLifecycleStage::ERROR;
        case PairLifecycleStage::VALIDATING:
            return target == PairLifecycleStage::CACHE_BUILDING || target == PairLifecycleStage::ERROR;
        case PairLifecycleStage::CACHE_BUILDING:
            return target == PairLifecycleStage::READY || target == PairLifecycleStage::ERROR;
        case PairLifecycleStage::READY:
            return target == PairLifecycleStage::TRADING || target == PairLifecycleStage::REMOVING;
        case PairLifecycleStage::TRADING:
            return target == PairLifecycleStage::PAUSING || target == PairLifecycleStage::REMOVING;
        case PairLifecycleStage::PAUSING:
            return target == PairLifecycleStage::READY;
        case PairLifecycleStage::REMOVING:
            return false; // Terminal state
        case PairLifecycleStage::ERROR:
            return target == PairLifecycleStage::INITIALIZING; // Can restart from error
        default:
            return false;
    }
}

bool DynamicPairManager::validatePairSymbolFormat(const std::string& symbol) const {
    // Validate currency pair format: CCC_CCC
    std::regex pattern(R"([A-Z]{3}_[A-Z]{3})");
    return std::regex_match(symbol, pattern);
}

bool DynamicPairManager::validateResourceRequirements(const DynamicPairConfig& config) const {
    ResourceAllocation usage = calculateCurrentUsage();
    usage.max_total_pairs += 1;
    usage.max_hot_bytes += config.required_hot_bytes;
    usage.max_streams += config.required_streams;
    usage.max_batch_size = std::max(usage.max_batch_size, config.batch_size);
    return validateResources(usage);
}

ResourceAllocation DynamicPairManager::calculateCurrentUsage() const {
    std::lock_guard<std::mutex> lock(config_mutex_);

    ResourceAllocation usage;
    usage.max_total_pairs = dynamic_configs_.size();

    // Count trading pairs
    for (const auto& stage_pair : pair_stages_) {
        if (stage_pair.second == PairLifecycleStage::TRADING) {
            usage.max_trading_pairs++;
        }
    }

    // Aggregate resource usage from active pair configurations
    size_t total_hot_bytes = 0;
    size_t total_streams = 0;
    size_t max_batch = 0;
    size_t max_hot_bytes_per_pair = 0;

    for (const auto& kv : dynamic_configs_) {
        const auto& cfg = kv.second;
        total_hot_bytes += cfg.required_hot_bytes;
        total_streams += cfg.required_streams;
        max_batch = std::max(max_batch, cfg.batch_size);
        max_hot_bytes_per_pair = std::max(max_hot_bytes_per_pair, cfg.required_hot_bytes);
    }

    usage.max_hot_bytes = total_hot_bytes;
    usage.max_streams = total_streams;
    usage.max_batch_size = max_batch;
    usage.max_memory_per_pair_mb =
        max_hot_bytes_per_pair > 0 ? max_hot_bytes_per_pair / (1024 * 1024) : 0;
    usage.cpu_limit_percentage = 0.0; // CPU usage tracking not yet implemented

    return usage;
}

bool DynamicPairManager::validateResources(const ResourceAllocation& usage) const {
    return usage.max_total_pairs <= resource_allocation_.max_total_pairs &&
           usage.max_memory_per_pair_mb <= resource_allocation_.max_memory_per_pair_mb &&
           usage.max_hot_bytes <= resource_allocation_.max_hot_bytes &&
           usage.max_streams <= resource_allocation_.max_streams &&
           usage.max_batch_size <= resource_allocation_.max_batch_size;
}

std::string DynamicPairManager::generateOperationId() {
    operation_counter_.fetch_add(1);
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    
    return "OP_" + std::to_string(timestamp) + "_" + std::to_string(operation_counter_.load());
}

PairOperation DynamicPairManager::createOperation(const std::string& pair_symbol, const std::string& operation_type) {
    PairOperation operation;
    operation.pair_symbol = pair_symbol;
    operation.operation_type = operation_type;
    operation.stage = PairLifecycleStage::INITIALIZING;
    operation.result = PairOperationResult::SUCCESS;
    operation.progress_percentage = 0.0;
    operation.start_time = std::chrono::system_clock::now();
    
    return operation;
}

void DynamicPairManager::updateOperationProgress(PairOperation& operation, double progress, const std::string& message) {
    operation.progress_percentage = progress;
    operation.status_message = message;
    operation.steps_completed.push_back(message);
    
    // Notify operation update
    notifyOperationUpdate(operation);
}

void DynamicPairManager::completeOperation(PairOperation& operation, PairOperationResult result) {
    operation.result = result;
    operation.end_time = std::chrono::system_clock::now();
    operation.duration = std::chrono::duration_cast<std::chrono::duration<double>>(
        operation.end_time - operation.start_time);
    
    // Update statistics
    total_operations_.fetch_add(1);
    total_operation_time_.store(total_operation_time_.load() + operation.duration);
    
    if (result == PairOperationResult::SUCCESS) {
        successful_operations_.fetch_add(1);
    } else {
        failed_operations_.fetch_add(1);
    }
    
    // Move to history
    {
        std::lock_guard<std::mutex> lock(operations_mutex_);
        operation_history_.push_back(operation);
        active_operations_.erase(operation.operation_id);
        
        // Maintain reasonable history size
        if (operation_history_.size() > 1000) {
            operation_history_.erase(operation_history_.begin());
        }
    }
    
    // Final notification
    notifyOperationUpdate(operation);
}

void DynamicPairManager::notifyLifecycleChange(const std::string& pair_symbol, PairLifecycleStage old_stage, PairLifecycleStage new_stage) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : lifecycle_callbacks_) {
        try {
            callback(pair_symbol, old_stage, new_stage);
        } catch (const std::exception&) {
            // Ignore callback errors
        }
    }
}

void DynamicPairManager::notifyOperationUpdate(const PairOperation& operation) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : operation_callbacks_) {
        try {
            callback(operation);
        } catch (const std::exception&) {
            // Ignore callback errors
        }
    }
}

// Utility functions
std::string pairOperationResultToString(PairOperationResult result) {
    switch (result) {
        case PairOperationResult::SUCCESS: return "SUCCESS";
        case PairOperationResult::ALREADY_EXISTS: return "ALREADY_EXISTS";
        case PairOperationResult::NOT_FOUND: return "NOT_FOUND";
        case PairOperationResult::INVALID_SYMBOL: return "INVALID_SYMBOL";
        case PairOperationResult::CACHE_NOT_READY: return "CACHE_NOT_READY";
        case PairOperationResult::TRADING_ACTIVE: return "TRADING_ACTIVE";
        case PairOperationResult::RESOURCE_LIMIT: return "RESOURCE_LIMIT";
        case PairOperationResult::VALIDATION_FAILED: return "VALIDATION_FAILED";
        case PairOperationResult::NETWORK_ERROR: return "NETWORK_ERROR";
        case PairOperationResult::CONFIGURATION_ERROR: return "CONFIGURATION_ERROR";
        case PairOperationResult::SYSTEM_ERROR: return "SYSTEM_ERROR";
        default: return "UNKNOWN";
    }
}

std::string pairLifecycleStageToString(PairLifecycleStage stage) {
    switch (stage) {
        case PairLifecycleStage::INITIALIZING: return "INITIALIZING";
        case PairLifecycleStage::VALIDATING: return "VALIDATING";
        case PairLifecycleStage::CACHE_BUILDING: return "CACHE_BUILDING";
        case PairLifecycleStage::READY: return "READY";
        case PairLifecycleStage::TRADING: return "TRADING";
        case PairLifecycleStage::PAUSING: return "PAUSING";
        case PairLifecycleStage::REMOVING: return "REMOVING";
        case PairLifecycleStage::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

bool isPairOperationResultSuccess(PairOperationResult result) {
    return result == PairOperationResult::SUCCESS;
}

bool isPairLifecycleStageActive(PairLifecycleStage stage) {
    return stage == PairLifecycleStage::READY || stage == PairLifecycleStage::TRADING;
}

std::vector<std::string> DynamicPairManager::getAllPairs() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    std::vector<std::string> pairs;
    pairs.reserve(dynamic_configs_.size());
    
    for (const auto& config_pair : dynamic_configs_) {
        pairs.push_back(config_pair.first);
    }
    
    return pairs;
}

bool DynamicPairManager::isPairEnabled(const std::string& pair) const {
    PairLifecycleStage stage = getPairLifecycleStage(pair);
    return stage == PairLifecycleStage::TRADING;
}

std::string DynamicPairManager::enablePairAsync(const std::string& pair_symbol) {
    if (operations_paused_.load()) {
        return ""; // Operations are paused
    }
    
    std::string operation_id = generateOperationId();
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    operation_queue_.push([this, pair_symbol, operation_id]() {
        std::lock_guard<std::mutex> ops_lock(operations_mutex_);
        
        PairOperation operation = createOperation(pair_symbol, "ENABLE");
        operation.operation_id = operation_id;
        
        active_operations_[operation_id] = operation;
        
        PairOperationResult result = performEnablePair(pair_symbol, active_operations_[operation_id]);
        completeOperation(active_operations_[operation_id], result);
    });
    
    operation_cv_.notify_one();
    return operation_id;
}

std::string DynamicPairManager::disablePairAsync(const std::string& pair_symbol) {
    if (operations_paused_.load()) {
        return ""; // Operations are paused
    }
    
    std::string operation_id = generateOperationId();
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    operation_queue_.push([this, pair_symbol, operation_id]() {
        std::lock_guard<std::mutex> ops_lock(operations_mutex_);
        
        PairOperation operation = createOperation(pair_symbol, "DISABLE");
        operation.operation_id = operation_id;
        
        active_operations_[operation_id] = operation;
        
        PairOperationResult result = performDisablePair(pair_symbol, active_operations_[operation_id]);
        completeOperation(active_operations_[operation_id], result);
    });
    
    operation_cv_.notify_one();
    return operation_id;
}

PairOperationResult DynamicPairManager::enablePair(const std::string& pair_symbol, std::chrono::seconds timeout) {
    std::string operation_id = enablePairAsync(pair_symbol);
    if (operation_id.empty()) {
        return PairOperationResult::SYSTEM_ERROR;
    }
    
    // Wait for completion
    auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start_time < timeout) {
        if (isOperationComplete(operation_id)) {
            return getOperation(operation_id).result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Timeout
    cancelOperation(operation_id);
    return PairOperationResult::SYSTEM_ERROR;
}

PairOperationResult DynamicPairManager::disablePair(const std::string& pair_symbol, std::chrono::seconds timeout) {
    std::string operation_id = disablePairAsync(pair_symbol);
    if (operation_id.empty()) {
        return PairOperationResult::SYSTEM_ERROR;
    }
    
    // Wait for completion
    auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start_time < timeout) {
        if (isOperationComplete(operation_id)) {
            return getOperation(operation_id).result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Timeout
    cancelOperation(operation_id);
    return PairOperationResult::SYSTEM_ERROR;
}

// Global dynamic pair manager instance
DynamicPairManager& getGlobalDynamicPairManager() {
    static DynamicPairManager instance;
    return instance;
}

} // namespace sep::trading
