#pragma once

#include "../core/pair_manager.hpp"
#include "../core/trading_state.hpp"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace sep::api {

// Response structure for API calls
struct ApiResponse {
    bool success;
    std::string message;
    std::string data;
    int error_code;
    
    ApiResponse() : success(false), error_code(0) {}
    ApiResponse(bool s, const std::string& msg, const std::string& d = "", int code = 0)
        : success(s), message(msg), data(d), error_code(code) {}
};

// State controller for API endpoints
class StateController {
public:
    StateController();
    ~StateController() = default;

    // Pair management endpoints
    ApiResponse getAllPairs() const;
    ApiResponse getPairInfo(const std::string& symbol) const;
    ApiResponse getPairsByStatus(const std::string& status) const;
    ApiResponse addPair(const std::string& symbol);
    ApiResponse removePair(const std::string& symbol);
    ApiResponse enablePair(const std::string& symbol);
    ApiResponse disablePair(const std::string& symbol);
    
    // Training management endpoints
    ApiResponse startTraining(const std::string& symbol);
    ApiResponse stopTraining(const std::string& symbol);
    ApiResponse getTrainingStatus(const std::string& symbol) const;
    ApiResponse updateModelInfo(const std::string& symbol, const std::string& model_path, double accuracy);
    
    // Trading control endpoints
    ApiResponse startTrading(const std::string& symbol);
    ApiResponse stopTrading(const std::string& symbol);
    ApiResponse getTradingStatus(const std::string& symbol) const;
    ApiResponse pauseAllTrading();
    ApiResponse resumeAllTrading();
    ApiResponse emergencyStop();
    
    // System status endpoints
    ApiResponse getSystemStatus() const;
    ApiResponse getSystemHealth() const;
    ApiResponse getSystemMetrics() const;
    ApiResponse setSystemStatus(const std::string& status);
    ApiResponse setMarketCondition(const std::string& condition);
    
    // Configuration endpoints
    ApiResponse setGlobalTradingEnabled(bool enabled);
    ApiResponse setMaintenanceMode(bool enabled);
    ApiResponse setRiskLevel(double level);
    ApiResponse setMaxPositions(size_t max_positions);
    ApiResponse reloadConfiguration();
    
    // Statistics endpoints
    ApiResponse getPairStatistics() const;
    ApiResponse getSystemStatistics() const;
    ApiResponse getPerformanceMetrics() const;
    
    // Error handling
    ApiResponse setPairError(const std::string& symbol, const std::string& error_message);
    ApiResponse clearPairError(const std::string& symbol);
    ApiResponse getSystemErrors() const;
    
    // State persistence
    ApiResponse saveState();
    ApiResponse loadState();
    ApiResponse exportState() const;
    ApiResponse importState(const std::string& state_data);
    
private:
    std::unique_ptr<sep::core::PairManager> pair_manager_;
    sep::core::TradingState& trading_state_;
    
    // Helper methods
    std::string formatPairInfo(const sep::core::PairInfo& info) const;
    std::string formatSystemHealth(const sep::core::SystemHealth& health) const;
    std::string formatSystemMetrics() const;
    std::string formatPairStatistics() const;
    std::string formatSystemStatistics() const;
    std::string formatPerformanceMetrics() const;
    
    // Validation helpers
    bool validatePairSymbol(const std::string& symbol) const;
    bool validateStatusString(const std::string& status) const;
    bool validateMarketCondition(const std::string& condition) const;
    bool validateRiskLevel(double level) const;
    
    // JSON formatting helpers
    std::string toJson(const std::vector<std::string>& items) const;
    std::string toJson(const sep::core::PairInfo& info) const;
    std::string toJson(const sep::core::SystemHealth& health) const;
    std::string escapeJsonString(const std::string& str) const;
    
    // Error code constants
    static constexpr int ERROR_PAIR_NOT_FOUND = 1001;
    static constexpr int ERROR_INVALID_STATUS = 1002;
    static constexpr int ERROR_INVALID_PARAMETER = 1003;
    static constexpr int ERROR_OPERATION_FAILED = 1004;
    static constexpr int ERROR_SYSTEM_ERROR = 1005;
    static constexpr int ERROR_PERMISSION_DENIED = 1006;
    static constexpr int ERROR_INVALID_STATE = 1007;
};

// Factory function for creating state controller
std::unique_ptr<StateController> createStateController();

// Utility functions for API responses
std::string formatApiResponse(const ApiResponse& response);
ApiResponse parseApiRequest(const std::string& request);

} // namespace sep::api
