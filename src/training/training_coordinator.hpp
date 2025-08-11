// SEP Professional Training Coordinator
// Coordinates local CUDA training with remote trading deployment
// Part of distributed pattern processing architecture

#ifndef TRAINING_COORDINATOR_HPP
#define TRAINING_COORDINATOR_HPP

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "config/dynamic_config_manager.hpp"
#include "cache/weekly_cache_manager.hpp"
#include "remote_synchronizer.hpp"
#include "weekly_data_fetcher.hpp"

namespace sep {
namespace training {

enum class TrainingMode {
    QUICK,          // Fast training for development
    FULL,           // Complete training with optimization
    LIVE_TUNE,      // Live parameter tuning for active trader
    BATCH           // Batch processing for multiple pairs
};

enum class PatternQuality {
    HIGH,           // >70% accuracy, ready for live trading
    MEDIUM,         // 60-70% accuracy, suitable for testing
    LOW,            // <60% accuracy, needs retraining
    UNKNOWN         // Not yet evaluated
};

struct TrainingResult {
    std::string pair;
    double accuracy;
    double stability_score;
    double coherence_score;
    double entropy_score;
    PatternQuality quality;
    std::chrono::system_clock::time_point trained_at;
    std::string model_hash;
    std::map<std::string, double> parameters;
};

struct RemoteTraderConfig {
    std::string host;                   // Tailscale IP (100.85.55.105)
    int port;                          // Remote API port
    std::string auth_token;            // Authentication token
    bool ssl_enabled;                  // HTTPS support
    std::chrono::seconds sync_interval; // How often to sync patterns
};

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
    // Core components
    std::unique_ptr<config::DynamicConfigManager> config_manager_;
    std::unique_ptr<cache::WeeklyCacheManager> cache_manager_;
    std::unique_ptr<WeeklyDataFetcher> data_fetcher_;
    std::unique_ptr<RemoteSynchronizer> remote_synchronizer_;
    
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
