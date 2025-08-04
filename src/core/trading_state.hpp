#pragma once

#include <atomic>
#include <string>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <functional>
#include <regex>

namespace sep::core {

enum class SystemStatus {
    INITIALIZING,   // System starting up
    IDLE,          // System ready but not trading
    TRADING,       // Active trading in progress
    PAUSED,        // Trading paused (manual or automatic)
    STOPPING,      // Graceful shutdown in progress
    ERROR,         // System error requiring attention
    MAINTENANCE    // Maintenance mode
};

enum class MarketCondition {
    UNKNOWN,       // Market condition not determined
    NORMAL,        // Normal trading conditions
    HIGH_VOLATILITY, // High volatility detected
    LOW_LIQUIDITY, // Low liquidity conditions
    NEWS_EVENT,    // Major news event detected
    MARKET_CLOSE   // Market closed
};

struct SystemHealth {
    double cpu_usage;
    double memory_usage;
    double network_latency;
    size_t active_connections;
    size_t pending_orders;
    std::chrono::system_clock::time_point last_update;
    
    SystemHealth() : cpu_usage(0.0), memory_usage(0.0), network_latency(0.0), 
                    active_connections(0), pending_orders(0), 
                    last_update(std::chrono::system_clock::now()) {}
};

// Global state change callback type
using SystemStateCallback = std::function<void(SystemStatus old_status, SystemStatus new_status)>;

class TradingState {
public:
    static TradingState& getInstance();
    
    // System status management
    bool setSystemStatus(SystemStatus status);
    SystemStatus getSystemStatus() const;
    std::string getSystemStatusString() const;
    
    // Market condition management
    bool setMarketCondition(MarketCondition condition);
    MarketCondition getMarketCondition() const;
    std::string getMarketConditionString() const;
    
    // Emergency controls
    bool emergencyStop();
    bool pauseTrading();
    bool resumeTrading();
    bool isEmergencyStopped() const;
    bool isTradingPaused() const;
    
    // Health monitoring
    void updateSystemHealth(const SystemHealth& health);
    SystemHealth getSystemHealth() const;
    bool isSystemHealthy() const;
    
    // Global trading flags
    void setGlobalTradingEnabled(bool enabled);
    bool isGlobalTradingEnabled() const;
    void setMaintenanceMode(bool enabled);
    bool isMaintenanceMode() const;
    
    // Risk management
    void setRiskLevel(double level); // 0.0 = no risk, 1.0 = maximum risk
    double getRiskLevel() const;
    void setMaxPositions(size_t max_positions);
    size_t getMaxPositions() const;
    
    // System metrics
    void incrementTradeCount();
    void incrementErrorCount();
    size_t getTotalTrades() const;
    size_t getTotalErrors() const;
    double getSuccessRate() const;
    
    // Timestamp tracking
    std::chrono::system_clock::time_point getStartupTime() const;
    std::chrono::system_clock::time_point getLastTradeTime() const;
    std::chrono::duration<double> getUptime() const;
    
    // Event system
    void addSystemStateCallback(SystemStateCallback callback);
    void removeSystemStateCallback(size_t callback_id);
    
    // Configuration
    void setConfigurationValid(bool valid);
    bool isConfigurationValid() const;
    void setLastConfigUpdate(std::chrono::system_clock::time_point time);
    std::chrono::system_clock::time_point getLastConfigUpdate() const;
    
    // Persistence
    bool saveState();
    bool loadState();
    
private:
    TradingState();
    ~TradingState() = default;
    
    // Prevent copying
    TradingState(const TradingState&) = delete;
    TradingState& operator=(const TradingState&) = delete;
    
    // Core state variables
    std::atomic<SystemStatus> system_status_{SystemStatus::INITIALIZING};
    std::atomic<MarketCondition> market_condition_{MarketCondition::UNKNOWN};
    std::atomic<bool> emergency_stop_{false};
    std::atomic<bool> trading_paused_{false};
    std::atomic<bool> global_trading_enabled_{false};
    std::atomic<bool> maintenance_mode_{false};
    std::atomic<bool> configuration_valid_{false};
    
    // Risk and limits
    std::atomic<double> risk_level_{0.5};
    std::atomic<size_t> max_positions_{10};
    
    // Metrics
    std::atomic<size_t> total_trades_{0};
    std::atomic<size_t> total_errors_{0};
    std::atomic<size_t> successful_trades_{0};
    
    // Timestamps
    std::chrono::system_clock::time_point startup_time_;
    std::atomic<std::chrono::system_clock::time_point> last_trade_time_;
    std::atomic<std::chrono::system_clock::time_point> last_config_update_;
    
    // Health data
    mutable std::mutex health_mutex_;
    SystemHealth current_health_;
    
    // Event callbacks
    std::vector<SystemStateCallback> state_callbacks_;
    std::mutex callbacks_mutex_;
    
    // Helper methods
    void notifySystemStateChange(SystemStatus old_status, SystemStatus new_status);
    std::string serializeState() const;
    bool deserializeState(const std::string& json_data);
};

// Utility functions
std::string systemStatusToString(SystemStatus status);
SystemStatus stringToSystemStatus(const std::string& status_str);
std::string marketConditionToString(MarketCondition condition);
MarketCondition stringToMarketCondition(const std::string& condition_str);

// Global access functions
TradingState& getGlobalTradingState();
bool isSystemReady();
bool canStartTrading();

} // namespace sep::core
