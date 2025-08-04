#include "state_controller.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

namespace sep::api {

StateController::StateController() 
    : pair_manager_(std::make_unique<sep::core::PairManager>()),
      trading_state_(sep::core::TradingState::getInstance()) {
}

ApiResponse StateController::getAllPairs() const {
    try {
        auto pairs = pair_manager_->getAllPairs();
        std::string json_data = toJson(pairs);
        return ApiResponse(true, "Successfully retrieved all pairs", json_data);
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to retrieve pairs: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::getPairInfo(const std::string& symbol) const {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        const auto& info = pair_manager_->getPairInfo(symbol);
        if (info.status == sep::core::PairStatus::ERROR && info.error_message == "Pair not found") {
            return ApiResponse(false, "Pair not found", "", ERROR_PAIR_NOT_FOUND);
        }
        
        std::string json_data = toJson(info);
        return ApiResponse(true, "Successfully retrieved pair info", json_data);
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to retrieve pair info: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::getPairsByStatus(const std::string& status) const {
    if (!validateStatusString(status)) {
        return ApiResponse(false, "Invalid status", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        sep::core::PairStatus pair_status = sep::core::stringToStatus(status);
        auto pairs = pair_manager_->getPairsByStatus(pair_status);
        std::string json_data = toJson(pairs);
        return ApiResponse(true, "Successfully retrieved pairs by status", json_data);
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to retrieve pairs by status: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::addPair(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        bool success = pair_manager_->addPair(symbol);
        if (success) {
            return ApiResponse(true, "Pair added successfully");
        } else {
            return ApiResponse(false, "Failed to add pair (may already exist)", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to add pair: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::removePair(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        bool success = pair_manager_->removePair(symbol);
        if (success) {
            return ApiResponse(true, "Pair removed successfully");
        } else {
            return ApiResponse(false, "Failed to remove pair (may not exist or currently trading)", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to remove pair: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::enablePair(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        bool success = pair_manager_->enablePair(symbol);
        if (success) {
            return ApiResponse(true, "Pair enabled successfully");
        } else {
            return ApiResponse(false, "Failed to enable pair (may not exist or not ready)", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to enable pair: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::disablePair(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        bool success = pair_manager_->disablePair(symbol);
        if (success) {
            return ApiResponse(true, "Pair disabled successfully");
        } else {
            return ApiResponse(false, "Failed to disable pair", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to disable pair: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::startTraining(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        bool success = pair_manager_->setPairStatus(symbol, sep::core::PairStatus::TRAINING);
        if (success) {
            return ApiResponse(true, "Training started successfully");
        } else {
            return ApiResponse(false, "Failed to start training", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to start training: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::stopTraining(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        auto current_status = pair_manager_->getPairStatus(symbol);
        if (current_status != sep::core::PairStatus::TRAINING) {
            return ApiResponse(false, "Pair is not currently training", "", ERROR_INVALID_STATE);
        }
        
        bool success = pair_manager_->setPairStatus(symbol, sep::core::PairStatus::UNTRAINED);
        if (success) {
            return ApiResponse(true, "Training stopped successfully");
        } else {
            return ApiResponse(false, "Failed to stop training", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to stop training: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::getTrainingStatus(const std::string& symbol) const {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        const auto& info = pair_manager_->getPairInfo(symbol);
        std::stringstream ss;
        ss << "{\n";
        ss << "  \"symbol\": \"" << symbol << "\",\n";
        ss << "  \"status\": \"" << sep::core::statusToString(info.status) << "\",\n";
        ss << "  \"is_training\": " << (info.status == sep::core::PairStatus::TRAINING ? "true" : "false") << "\n";
        ss << "}";
        
        return ApiResponse(true, "Successfully retrieved training status", ss.str());
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to get training status: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::updateModelInfo(const std::string& symbol, const std::string& model_path, double accuracy) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    if (accuracy < 0.0 || accuracy > 1.0) {
        return ApiResponse(false, "Invalid accuracy value (must be 0.0-1.0)", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        bool success = pair_manager_->updateModel(symbol, model_path, accuracy);
        if (success) {
            return ApiResponse(true, "Model info updated successfully");
        } else {
            return ApiResponse(false, "Failed to update model info", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to update model info: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::startTrading(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        // Check if global trading is enabled
        if (!trading_state_.isGlobalTradingEnabled()) {
            return ApiResponse(false, "Global trading is disabled", "", ERROR_PERMISSION_DENIED);
        }
        
        if (!sep::core::canStartTrading()) {
            return ApiResponse(false, "System is not ready for trading", "", ERROR_INVALID_STATE);
        }
        
        bool success = pair_manager_->startTrading(symbol);
        if (success) {
            return ApiResponse(true, "Trading started successfully");
        } else {
            return ApiResponse(false, "Failed to start trading (pair may not be ready)", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to start trading: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::stopTrading(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        bool success = pair_manager_->stopTrading(symbol);
        if (success) {
            return ApiResponse(true, "Trading stopped successfully");
        } else {
            return ApiResponse(false, "Failed to stop trading", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to stop trading: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::getTradingStatus(const std::string& symbol) const {
    if (!validatePairSymbol(symbol)) {
        return ApiResponse(false, "Invalid pair symbol", "", ERROR_INVALID_PARAMETER);
    }
    
    try {
        bool is_trading = pair_manager_->isTrading(symbol);
        const auto& info = pair_manager_->getPairInfo(symbol);
        
        std::stringstream ss;
        ss << "{\n";
        ss << "  \"symbol\": \"" << symbol << "\",\n";
        ss << "  \"is_trading\": " << (is_trading ? "true" : "false") << ",\n";
        ss << "  \"status\": \"" << sep::core::statusToString(info.status) << "\",\n";
        ss << "  \"enabled\": " << (info.enabled ? "true" : "false") << "\n";
        ss << "}";
        
        return ApiResponse(true, "Successfully retrieved trading status", ss.str());
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to get trading status: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::pauseAllTrading() {
    try {
        bool success = trading_state_.pauseTrading();
        if (success) {
            return ApiResponse(true, "All trading paused successfully");
        } else {
            return ApiResponse(false, "Failed to pause trading", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to pause trading: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::resumeAllTrading() {
    try {
        bool success = trading_state_.resumeTrading();
        if (success) {
            return ApiResponse(true, "Trading resumed successfully");
        } else {
            return ApiResponse(false, "Failed to resume trading (may be in emergency stop)", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to resume trading: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::emergencyStop() {
    try {
        bool success = trading_state_.emergencyStop();
        if (success) {
            return ApiResponse(true, "Emergency stop activated - all trading halted");
        } else {
            return ApiResponse(false, "Failed to activate emergency stop", "", ERROR_OPERATION_FAILED);
        }
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to activate emergency stop: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::getSystemStatus() const {
    try {
        std::stringstream ss;
        ss << "{\n";
        ss << "  \"system_status\": \"" << trading_state_.getSystemStatusString() << "\",\n";
        ss << "  \"market_condition\": \"" << trading_state_.getMarketConditionString() << "\",\n";
        ss << "  \"global_trading_enabled\": " << (trading_state_.isGlobalTradingEnabled() ? "true" : "false") << ",\n";
        ss << "  \"trading_paused\": " << (trading_state_.isTradingPaused() ? "true" : "false") << ",\n";
        ss << "  \"emergency_stop\": " << (trading_state_.isEmergencyStopped() ? "true" : "false") << ",\n";
        ss << "  \"maintenance_mode\": " << (trading_state_.isMaintenanceMode() ? "true" : "false") << ",\n";
        ss << "  \"configuration_valid\": " << (trading_state_.isConfigurationValid() ? "true" : "false") << ",\n";
        ss << "  \"system_healthy\": " << (trading_state_.isSystemHealthy() ? "true" : "false") << "\n";
        ss << "}";
        
        return ApiResponse(true, "Successfully retrieved system status", ss.str());
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to get system status: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::getSystemHealth() const {
    try {
        auto health = trading_state_.getSystemHealth();
        std::string json_data = toJson(health);
        return ApiResponse(true, "Successfully retrieved system health", json_data);
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to get system health: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

ApiResponse StateController::getSystemMetrics() const {
    try {
        std::string json_data = formatSystemMetrics();
        return ApiResponse(true, "Successfully retrieved system metrics", json_data);
    } catch (const std::exception& e) {
        return ApiResponse(false, "Failed to get system metrics: " + std::string(e.what()), "", ERROR_SYSTEM_ERROR);
    }
}

// Helper methods implementation
std::string StateController::toJson(const std::vector<std::string>& items) const {
    std::stringstream ss;
    ss << "[\n";
    for (size_t i = 0; i < items.size(); ++i) {
        ss << "  \"" << escapeJsonString(items[i]) << "\"";
        if (i < items.size() - 1) ss << ",";
        ss << "\n";
    }
    ss << "]";
    return ss.str();
}

std::string StateController::toJson(const sep::core::PairInfo& info) const {
    std::stringstream ss;
    auto last_updated = std::chrono::system_clock::to_time_t(info.last_updated);
    
    ss << "{\n";
    ss << "  \"symbol\": \"" << escapeJsonString(info.symbol) << "\",\n";
    ss << "  \"status\": \"" << sep::core::statusToString(info.status) << "\",\n";
    ss << "  \"enabled\": " << (info.enabled ? "true" : "false") << ",\n";
    ss << "  \"trading_active\": " << (info.trading_active.load() ? "true" : "false") << ",\n";
    ss << "  \"accuracy\": " << info.accuracy << ",\n";
    ss << "  \"model_path\": \"" << escapeJsonString(info.model_path) << "\",\n";
    ss << "  \"error_message\": \"" << escapeJsonString(info.error_message) << "\",\n";
    ss << "  \"last_updated\": " << last_updated << "\n";
    ss << "}";
    
    return ss.str();
}

std::string StateController::toJson(const sep::core::SystemHealth& health) const {
    std::stringstream ss;
    auto last_update = std::chrono::system_clock::to_time_t(health.last_update);
    
    ss << "{\n";
    ss << "  \"cpu_usage\": " << health.cpu_usage << ",\n";
    ss << "  \"memory_usage\": " << health.memory_usage << ",\n";
    ss << "  \"network_latency\": " << health.network_latency << ",\n";
    ss << "  \"active_connections\": " << health.active_connections << ",\n";
    ss << "  \"pending_orders\": " << health.pending_orders << ",\n";
    ss << "  \"last_update\": " << last_update << "\n";
    ss << "}";
    
    return ss.str();
}

std::string StateController::formatSystemMetrics() const {
    std::stringstream ss;
    auto uptime = trading_state_.getUptime();
    
    ss << "{\n";
    ss << "  \"total_pairs\": " << pair_manager_->getTotalPairs() << ",\n";
    ss << "  \"active_pairs\": " << pair_manager_->getActivePairs() << ",\n";
    ss << "  \"training_pairs\": " << pair_manager_->getTrainingPairs() << ",\n";
    ss << "  \"average_accuracy\": " << pair_manager_->getAverageAccuracy() << ",\n";
    ss << "  \"total_trades\": " << trading_state_.getTotalTrades() << ",\n";
    ss << "  \"total_errors\": " << trading_state_.getTotalErrors() << ",\n";
    ss << "  \"success_rate\": " << trading_state_.getSuccessRate() << ",\n";
    ss << "  \"uptime_seconds\": " << uptime.count() << ",\n";
    ss << "  \"risk_level\": " << trading_state_.getRiskLevel() << ",\n";
    ss << "  \"max_positions\": " << trading_state_.getMaxPositions() << "\n";
    ss << "}";
    
    return ss.str();
}

std::string StateController::escapeJsonString(const std::string& str) const {
    std::string escaped;
    for (char c : str) {
        switch (c) {
            case '\"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}

bool StateController::validatePairSymbol(const std::string& symbol) const {
    return sep::core::isValidPairSymbol(symbol);
}

bool StateController::validateStatusString(const std::string& status) const {
    return status == "UNTRAINED" || status == "TRAINING" || status == "READY" || 
           status == "TRADING" || status == "DISABLED" || status == "ERROR";
}

bool StateController::validateMarketCondition(const std::string& condition) const {
    return condition == "UNKNOWN" || condition == "NORMAL" || condition == "HIGH_VOLATILITY" ||
           condition == "LOW_LIQUIDITY" || condition == "NEWS_EVENT" || condition == "MARKET_CLOSE";
}

bool StateController::validateRiskLevel(double level) const {
    return level >= 0.0 && level <= 1.0;
}

// Factory function
std::unique_ptr<StateController> createStateController() {
    return std::make_unique<StateController>();
}

// Utility functions
std::string formatApiResponse(const ApiResponse& response) {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"success\": " << (response.success ? "true" : "false") << ",\n";
    ss << "  \"message\": \"" << response.message << "\",\n";
    ss << "  \"error_code\": " << response.error_code << ",\n";
    ss << "  \"data\": " << (response.data.empty() ? "null" : response.data) << "\n";
    ss << "}";
    return ss.str();
}

} // namespace sep::api
