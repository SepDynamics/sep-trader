// SEP Training CLI Commands
// Command implementations for the professional training interface

#ifndef CLI_COMMANDS_HPP
#define CLI_COMMANDS_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "core/training_coordinator.hpp"

// Forward declarations
namespace sep::train {
    class Orchestrator;
    struct TrainResult;
    enum class Quality : uint8_t;
    enum class Mode : uint8_t;
}

namespace sep {
namespace training {

// Legacy struct for compatibility
struct RemoteTraderConfig {
    std::string endpoint;
    std::string auth_token;
    int timeout_seconds = 30;
    bool enabled = false;
};

class CLICommands {
public:
    explicit CLICommands(sep::train::Orchestrator& coordinator);
    
    // Training operations
    bool trainPair(const std::string& pair);
    bool trainAllPairs(bool quick_mode = false);
    bool trainSelectedPairs(const std::string& pairs_csv);
    bool retrainFailedPairs();
    
    // Data management
    bool fetchWeeklyData();
    bool fetchWeeklyData(const std::string& pair);
    bool validateCache();
    bool cleanupCache();
    
    // Remote integration
    bool configureRemoteTrader(const std::string& remote_ip);
    bool syncPatternsToRemote();
    bool syncParametersFromRemote();
    bool testRemoteConnection();
    
    // Live tuning
    bool startLiveTuning(const std::string& pairs_csv);
    bool stopLiveTuning();
    
    // System operations
    bool runBenchmark();
    
private:
    sep::train::Orchestrator& coordinator_;
    
    // Utility methods
    std::vector<std::string> parsePairsList(const std::string& pairs_csv);
    void printProgress(const std::string& operation, int current, int total);
    void printTrainingResult(const std::string& pair, const sep::train::TrainResult& result);
    bool confirmOperation(const std::string& operation);
};

} // namespace training
} // namespace sep

#endif // CLI_COMMANDS_HPP
