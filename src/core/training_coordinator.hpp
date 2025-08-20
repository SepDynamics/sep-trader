// SEP Professional Training Coordinator
// Coordinates local CUDA training with remote trading deployment
// Part of distributed pattern processing architecture

#ifndef TRAINING_COORDINATOR_HPP
#define TRAINING_COORDINATOR_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "training_types.h"  // Include the types from training_types.h

namespace sep {

// Forward declarations
namespace config {
    class DynamicConfigManager;
}
namespace cache {
    class WeeklyCacheManager;
}

namespace training {

// Forward declarations for training namespace
class RemoteSynchronizer;
class WeeklyDataFetcher;

class TrainingCoordinator {
public:
    TrainingCoordinator();
    ~TrainingCoordinator();

    // Core training operations
    bool trainPair(const std::string& pair, TrainingMode mode = TrainingMode::FULL);
    bool trainAllPairs(TrainingMode mode = TrainingMode::FULL);
    bool trainSelected(const std::vector<std::string>& pairs, TrainingMode mode = TrainingMode::FULL);
    
    // Pattern management
    TrainingResult getTrainingResult(const std::string& pair) const;
    std::vector<TrainingResult> getAllResults() const;
    bool validatePattern(const std::string& pair) const;
    
    // Remote trader integration
    bool configureRemoteTrader(const RemoteTraderConfig& config);
    bool syncPatternsToRemote();
    bool syncParametersFromRemote();
    bool isRemoteTraderConnected() const;
    
    // Live tuning system
    bool startLiveTuning(const std::vector<std::string>& pairs);
    bool stopLiveTuning();
    bool isLiveTuningActive() const;
    
    // Status and monitoring
    std::map<std::string, std::string> getSystemStatus() const;
    double getOverallSystemAccuracy() const;
    std::vector<std::string> getReadyPairs() const;
    std::vector<std::string> getFailedPairs() const;
    
    // Weekly data management
    bool fetchWeeklyDataForAll();
    bool fetchWeeklyDataForPair(const std::string& pair);
    bool validateWeeklyCache() const;
    
    // Resource contribution system
    bool contributePattern(const std::string& pair, const TrainingResult& result);
    bool requestOptimalParameters(const std::string& pair);
    
private:
    // Core components - using raw pointers with forward declarations
    config::DynamicConfigManager* config_manager_;
    cache::WeeklyCacheManager* cache_manager_;
    WeeklyDataFetcher* data_fetcher_;
    RemoteSynchronizer* remote_synchronizer_;
    
    // Training state
    std::map<std::string, TrainingResult> training_results_;
    std::map<std::string, std::chrono::system_clock::time_point> last_trained_;
    mutable std::mutex results_mutex_;
    
    // Remote trader communication
    RemoteTraderConfig remote_config_;
    std::atomic<bool> remote_connected_;
    std::thread sync_thread_;
    std::atomic<bool> sync_running_;
    
    // Live tuning system
    std::atomic<bool> live_tuning_active_;
    std::thread tuning_thread_;
    std::queue<std::string> tuning_queue_;
    std::mutex tuning_mutex_;
    std::condition_variable tuning_cv_;
    
    // Internal methods
    bool initializeComponents();
    TrainingResult executeCudaTraining(const std::string& pair, TrainingMode mode);
    bool saveTrainingResult(const TrainingResult& result);
    bool loadTrainingResults();
    
    // Remote communication
    void syncThreadFunction();
    bool sendPatternToRemote(const std::string& pair, const TrainingResult& result);
    bool receiveParametersFromRemote(const std::string& pair);
    
    // Live tuning
    void liveTuningThreadFunction();
    bool performLiveTuning(const std::string& pair);
    
    // Utility methods
    PatternQuality assessPatternQuality(double accuracy) const;
    std::string generateModelHash(const TrainingResult& result) const;
    bool validateConfiguration() const;
};

} // namespace training
} // namespace sep

#endif // TRAINING_COORDINATOR_HPP
