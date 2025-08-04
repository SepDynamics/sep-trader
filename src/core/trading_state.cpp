#include "trading_state.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace sep::core {

TradingState::TradingState() 
    : startup_time_(std::chrono::system_clock::now()) {
    last_trade_time_.store(startup_time_);
    last_config_update_.store(startup_time_);
    loadState();
}

TradingState& TradingState::getInstance() {
    static TradingState instance;
    return instance;
}

bool TradingState::setSystemStatus(SystemStatus status) {
    SystemStatus old_status = system_status_.exchange(status);
    
    if (old_status != status) {
        notifySystemStateChange(old_status, status);
        saveState();
    }
    
    return true;
}

SystemStatus TradingState::getSystemStatus() const {
    return system_status_.load();
}

std::string TradingState::getSystemStatusString() const {
    return systemStatusToString(getSystemStatus());
}

bool TradingState::setMarketCondition(MarketCondition condition) {
    market_condition_.store(condition);
    saveState();
    return true;
}

MarketCondition TradingState::getMarketCondition() const {
    return market_condition_.load();
}

std::string TradingState::getMarketConditionString() const {
    return marketConditionToString(getMarketCondition());
}

bool TradingState::emergencyStop() {
    emergency_stop_.store(true);
    trading_paused_.store(true);
    setSystemStatus(SystemStatus::ERROR);
    
    std::cout << "EMERGENCY STOP ACTIVATED - All trading halted immediately!" << std::endl;
    return true;
}

bool TradingState::pauseTrading() {
    trading_paused_.store(true);
    setSystemStatus(SystemStatus::PAUSED);
    return true;
}

bool TradingState::resumeTrading() {
    if (emergency_stop_.load()) {
        return false; // Cannot resume from emergency stop without manual intervention
    }
    
    trading_paused_.store(false);
    setSystemStatus(SystemStatus::TRADING);
    return true;
}

bool TradingState::isEmergencyStopped() const {
    return emergency_stop_.load();
}

bool TradingState::isTradingPaused() const {
    return trading_paused_.load();
}

void TradingState::updateSystemHealth(const SystemHealth& health) {
    std::lock_guard<std::mutex> lock(health_mutex_);
    current_health_ = health;
    current_health_.last_update = std::chrono::system_clock::now();
}

SystemHealth TradingState::getSystemHealth() const {
    std::lock_guard<std::mutex> lock(health_mutex_);
    return current_health_;
}

bool TradingState::isSystemHealthy() const {
    std::lock_guard<std::mutex> lock(health_mutex_);
    
    // Define health thresholds
    const double MAX_CPU_USAGE = 90.0;
    const double MAX_MEMORY_USAGE = 85.0;
    const double MAX_NETWORK_LATENCY = 1000.0; // ms
    
    // Check if health data is recent (within last minute)
    auto now = std::chrono::system_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(now - current_health_.last_update);
    if (time_diff.count() > 60) {
        return false; // Stale health data
    }
    
    return current_health_.cpu_usage < MAX_CPU_USAGE &&
           current_health_.memory_usage < MAX_MEMORY_USAGE &&
           current_health_.network_latency < MAX_NETWORK_LATENCY;
}

void TradingState::setGlobalTradingEnabled(bool enabled) {
    global_trading_enabled_.store(enabled);
    
    if (!enabled) {
        pauseTrading();
    }
    
    saveState();
}

bool TradingState::isGlobalTradingEnabled() const {
    return global_trading_enabled_.load();
}

void TradingState::setMaintenanceMode(bool enabled) {
    maintenance_mode_.store(enabled);
    
    if (enabled) {
        setSystemStatus(SystemStatus::MAINTENANCE);
        pauseTrading();
    } else if (global_trading_enabled_.load()) {
        setSystemStatus(SystemStatus::IDLE);
    }
    
    saveState();
}

bool TradingState::isMaintenanceMode() const {
    return maintenance_mode_.load();
}

void TradingState::setRiskLevel(double level) {
    // Clamp risk level between 0.0 and 1.0
    level = std::max(0.0, std::min(1.0, level));
    risk_level_.store(level);
    saveState();
}

double TradingState::getRiskLevel() const {
    return risk_level_.load();
}

void TradingState::setMaxPositions(size_t max_positions) {
    max_positions_.store(max_positions);
    saveState();
}

size_t TradingState::getMaxPositions() const {
    return max_positions_.load();
}

void TradingState::incrementTradeCount() {
    total_trades_.fetch_add(1);
    last_trade_time_.store(std::chrono::system_clock::now());
}

void TradingState::incrementErrorCount() {
    total_errors_.fetch_add(1);
}

size_t TradingState::getTotalTrades() const {
    return total_trades_.load();
}

size_t TradingState::getTotalErrors() const {
    return total_errors_.load();
}

double TradingState::getSuccessRate() const {
    size_t total = total_trades_.load();
    size_t errors = total_errors_.load();
    
    if (total == 0) return 0.0;
    
    size_t successful = total - std::min(errors, total);
    return static_cast<double>(successful) / total * 100.0;
}

std::chrono::system_clock::time_point TradingState::getStartupTime() const {
    return startup_time_;
}

std::chrono::system_clock::time_point TradingState::getLastTradeTime() const {
    return last_trade_time_.load();
}

std::chrono::duration<double> TradingState::getUptime() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now - startup_time_);
}

void TradingState::addSystemStateCallback(SystemStateCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    state_callbacks_.push_back(callback);
}

void TradingState::removeSystemStateCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < state_callbacks_.size()) {
        state_callbacks_.erase(state_callbacks_.begin() + callback_id);
    }
}

void TradingState::setConfigurationValid(bool valid) {
    configuration_valid_.store(valid);
    last_config_update_.store(std::chrono::system_clock::now());
    saveState();
}

bool TradingState::isConfigurationValid() const {
    return configuration_valid_.load();
}

void TradingState::setLastConfigUpdate(std::chrono::system_clock::time_point time) {
    last_config_update_.store(time);
}

std::chrono::system_clock::time_point TradingState::getLastConfigUpdate() const {
    return last_config_update_.load();
}

bool TradingState::saveState() {
    try {
        std::string json_data = serializeState();
        std::ofstream file("/sep/config/trading_state.json");
        if (!file.is_open()) {
            return false;
        }
        file << json_data;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving trading state: " << e.what() << std::endl;
        return false;
    }
}

bool TradingState::loadState() {
    try {
        std::ifstream file("/sep/config/trading_state.json");
        if (!file.is_open()) {
            return false; // File doesn't exist yet, use defaults
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_data = buffer.str();
        
        return deserializeState(json_data);
    } catch (const std::exception& e) {
        std::cerr << "Error loading trading state: " << e.what() << std::endl;
        return false;
    }
}

void TradingState::notifySystemStateChange(SystemStatus old_status, SystemStatus new_status) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : state_callbacks_) {
        try {
            callback(old_status, new_status);
        } catch (const std::exception& e) {
            std::cerr << "Error in system state callback: " << e.what() << std::endl;
        }
    }
}

std::string TradingState::serializeState() const {
    std::stringstream ss;
    
    auto system_status = system_status_.load();
    auto market_condition = market_condition_.load();
    auto emergency_stop = emergency_stop_.load();
    auto trading_paused = trading_paused_.load();
    auto global_trading_enabled = global_trading_enabled_.load();
    auto maintenance_mode = maintenance_mode_.load();
    auto configuration_valid = configuration_valid_.load();
    auto risk_level = risk_level_.load();
    auto max_positions = max_positions_.load();
    auto total_trades = total_trades_.load();
    auto total_errors = total_errors_.load();
    
    auto startup_time_t = std::chrono::system_clock::to_time_t(startup_time_);
    auto last_trade_time_t = std::chrono::system_clock::to_time_t(last_trade_time_.load());
    auto last_config_time_t = std::chrono::system_clock::to_time_t(last_config_update_.load());
    
    ss << "{\n";
    ss << "  \"system_status\": \"" << systemStatusToString(system_status) << "\",\n";
    ss << "  \"market_condition\": \"" << marketConditionToString(market_condition) << "\",\n";
    ss << "  \"emergency_stop\": " << (emergency_stop ? "true" : "false") << ",\n";
    ss << "  \"trading_paused\": " << (trading_paused ? "true" : "false") << ",\n";
    ss << "  \"global_trading_enabled\": " << (global_trading_enabled ? "true" : "false") << ",\n";
    ss << "  \"maintenance_mode\": " << (maintenance_mode ? "true" : "false") << ",\n";
    ss << "  \"configuration_valid\": " << (configuration_valid ? "true" : "false") << ",\n";
    ss << "  \"risk_level\": " << risk_level << ",\n";
    ss << "  \"max_positions\": " << max_positions << ",\n";
    ss << "  \"total_trades\": " << total_trades << ",\n";
    ss << "  \"total_errors\": " << total_errors << ",\n";
    ss << "  \"startup_time\": " << startup_time_t << ",\n";
    ss << "  \"last_trade_time\": " << last_trade_time_t << ",\n";
    ss << "  \"last_config_update\": " << last_config_time_t << "\n";
    ss << "}";
    
    return ss.str();
}

bool TradingState::deserializeState(const std::string& json_data) {
    // Simple JSON parsing - would use proper JSON library in production
    try {
        // Extract key values using simple string operations
        // This is a minimal implementation for demonstration
        
        if (json_data.find("\"emergency_stop\": true") != std::string::npos) {
            emergency_stop_.store(true);
        }
        
        if (json_data.find("\"trading_paused\": true") != std::string::npos) {
            trading_paused_.store(true);
        }
        
        if (json_data.find("\"global_trading_enabled\": true") != std::string::npos) {
            global_trading_enabled_.store(true);
        }
        
        if (json_data.find("\"maintenance_mode\": true") != std::string::npos) {
            maintenance_mode_.store(true);
        }
        
        if (json_data.find("\"configuration_valid\": true") != std::string::npos) {
            configuration_valid_.store(true);
        }
        
        // Parse numeric values
        std::regex risk_regex(R"("risk_level":\s*([0-9.]+))");
        std::smatch match;
        if (std::regex_search(json_data, match, risk_regex)) {
            risk_level_.store(std::stod(match[1].str()));
        }
        
        std::regex positions_regex(R"("max_positions":\s*(\d+))");
        if (std::regex_search(json_data, match, positions_regex)) {
            max_positions_.store(std::stoull(match[1].str()));
        }
        
        std::regex trades_regex(R"("total_trades":\s*(\d+))");
        if (std::regex_search(json_data, match, trades_regex)) {
            total_trades_.store(std::stoull(match[1].str()));
        }
        
        std::regex errors_regex(R"("total_errors":\s*(\d+))");
        if (std::regex_search(json_data, match, errors_regex)) {
            total_errors_.store(std::stoull(match[1].str()));
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing trading state JSON: " << e.what() << std::endl;
        return false;
    }
}

// Utility functions
std::string systemStatusToString(SystemStatus status) {
    switch (status) {
        case SystemStatus::INITIALIZING: return "INITIALIZING";
        case SystemStatus::IDLE: return "IDLE";
        case SystemStatus::TRADING: return "TRADING";
        case SystemStatus::PAUSED: return "PAUSED";
        case SystemStatus::STOPPING: return "STOPPING";
        case SystemStatus::ERROR: return "ERROR";
        case SystemStatus::MAINTENANCE: return "MAINTENANCE";
        default: return "UNKNOWN";
    }
}

SystemStatus stringToSystemStatus(const std::string& status_str) {
    if (status_str == "INITIALIZING") return SystemStatus::INITIALIZING;
    if (status_str == "IDLE") return SystemStatus::IDLE;
    if (status_str == "TRADING") return SystemStatus::TRADING;
    if (status_str == "PAUSED") return SystemStatus::PAUSED;
    if (status_str == "STOPPING") return SystemStatus::STOPPING;
    if (status_str == "ERROR") return SystemStatus::ERROR;
    if (status_str == "MAINTENANCE") return SystemStatus::MAINTENANCE;
    return SystemStatus::ERROR;
}

std::string marketConditionToString(MarketCondition condition) {
    switch (condition) {
        case MarketCondition::UNKNOWN: return "UNKNOWN";
        case MarketCondition::NORMAL: return "NORMAL";
        case MarketCondition::HIGH_VOLATILITY: return "HIGH_VOLATILITY";
        case MarketCondition::LOW_LIQUIDITY: return "LOW_LIQUIDITY";
        case MarketCondition::NEWS_EVENT: return "NEWS_EVENT";
        case MarketCondition::MARKET_CLOSE: return "MARKET_CLOSE";
        default: return "UNKNOWN";
    }
}

MarketCondition stringToMarketCondition(const std::string& condition_str) {
    if (condition_str == "UNKNOWN") return MarketCondition::UNKNOWN;
    if (condition_str == "NORMAL") return MarketCondition::NORMAL;
    if (condition_str == "HIGH_VOLATILITY") return MarketCondition::HIGH_VOLATILITY;
    if (condition_str == "LOW_LIQUIDITY") return MarketCondition::LOW_LIQUIDITY;
    if (condition_str == "NEWS_EVENT") return MarketCondition::NEWS_EVENT;
    if (condition_str == "MARKET_CLOSE") return MarketCondition::MARKET_CLOSE;
    return MarketCondition::UNKNOWN;
}

// Global access functions
TradingState& getGlobalTradingState() {
    return TradingState::getInstance();
}

bool isSystemReady() {
    auto& state = TradingState::getInstance();
    auto status = state.getSystemStatus();
    
    return status == SystemStatus::IDLE || 
           status == SystemStatus::TRADING ||
           status == SystemStatus::PAUSED;
}

bool canStartTrading() {
    auto& state = TradingState::getInstance();
    
    return state.isGlobalTradingEnabled() &&
           !state.isEmergencyStopped() &&
           !state.isTradingPaused() &&
           !state.isMaintenanceMode() &&
           state.isConfigurationValid() &&
           state.isSystemHealthy();
}

} // namespace sep::core
