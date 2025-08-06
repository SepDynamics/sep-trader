#pragma once
#include "engine/internal/standard_includes.h"
#include "trading_state.hpp"

namespace sep::core {

enum class PairStatus {
    UNTRAINED,      // Pair has no trained model
    TRAINING,       // Currently training
    READY,          // Trained and ready to trade
    TRADING,        // Currently active in trading
    DISABLED,       // Manually disabled
    ERROR           // Error state requiring attention
};

struct PairInfo {
    std::string symbol;
    PairStatus status;
    std::chrono::system_clock::time_point last_trained;
    std::chrono::system_clock::time_point last_updated;
    double accuracy;
    std::string model_path;
    std::string error_message;
    bool enabled;
    std::atomic<bool> trading_active{false};
    
    PairInfo() : status(PairStatus::UNTRAINED), accuracy(0.0), enabled(false) {}
};

// State change event callback type
using StateChangeCallback = std::function<void(const std::string& symbol, PairStatus old_status, PairStatus new_status)>;

class PairManager {
public:
    PairManager();
    ~PairManager();

    // Core pair management
    bool addPair(const std::string& symbol);
    bool removePair(const std::string& symbol);
    bool enablePair(const std::string& symbol);
    bool disablePair(const std::string& symbol);
    
    // Status management
    bool setPairStatus(const std::string& symbol, PairStatus status);
    PairStatus getPairStatus(const std::string& symbol) const;
    const PairInfo& getPairInfo(const std::string& symbol) const;
    std::vector<std::string> getAllPairs() const;
    std::vector<std::string> getPairsByStatus(PairStatus status) const;
    
    // Trading lifecycle
    bool startTrading(const std::string& symbol);
    bool stopTrading(const std::string& symbol);
    bool isTrading(const std::string& symbol) const;
    
    // Model management
    bool updateModel(const std::string& symbol, const std::string& model_path, double accuracy);
    bool validateModel(const std::string& symbol) const;
    
    // Error handling
    bool setError(const std::string& symbol, const std::string& error_message);
    bool clearError(const std::string& symbol);
    
    // Event system
    void addStateChangeCallback(StateChangeCallback callback);
    void removeStateChangeCallback(size_t callback_id);
    
    // Persistence
    bool saveState();
    bool loadState();
    
    // Thread safety
    std::shared_lock<std::shared_mutex> getReadLock() const;
    std::unique_lock<std::shared_mutex> getWriteLock();
    
    // Statistics
    size_t getTotalPairs() const;
    size_t getActivePairs() const;
    size_t getTrainingPairs() const;
    double getAverageAccuracy() const;
    
private:
    mutable std::shared_mutex pairs_mutex_;
    std::unordered_map<std::string, std::unique_ptr<PairInfo>> pairs_;
    std::vector<StateChangeCallback> state_callbacks_;
    std::mutex callbacks_mutex_;
    std::string state_file_path_;
    
    // Internal helpers
    void notifyStateChange(const std::string& symbol, PairStatus old_status, PairStatus new_status);
    bool validatePairSymbol(const std::string& symbol) const;
    void initializeDefaultPairs();
    
    // JSON serialization helpers
    std::string serializeState() const;
    bool deserializeState(const std::string& json_data);
};

// Utility functions
std::string statusToString(PairStatus status);
PairStatus stringToStatus(const std::string& status_str);
bool isValidPairSymbol(const std::string& symbol);

} // namespace sep::core
