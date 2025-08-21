// SEP Training Status Display
// Professional status and monitoring display for training system

#ifndef STATUS_DISPLAY_HPP
#define STATUS_DISPLAY_HPP

#include <string>
#include <map>
#include <vector>
#include <chrono>

// Forward declarations
namespace sep::train {
    class Orchestrator;
    struct TrainResult;
    enum class Quality : uint8_t;
}

namespace sep {

class StatusDisplay {
public:
    explicit StatusDisplay(sep::train::Orchestrator& coordinator);
    
    // Status displays
    bool showSystemStatus();
    bool showSystemHealth();
    bool showTuningStatus();
    bool startMonitoringMode(int duration_seconds);
    
private:
    sep::train::Orchestrator& coordinator_;
    
    // Display utilities
    void printStatusHeader(const std::string& title);
    void printStatusLine(const std::string& label, const std::string& value, bool good = true);
    void printTrainingTable(const std::vector<sep::train::TrainResult>& results);
    void printRemoteStatus();
    void printCacheStatus();
    void printPerformanceMetrics();
    
    // Formatting utilities
    std::string formatAccuracy(double accuracy);
    std::string formatDuration(std::chrono::system_clock::time_point timestamp);
    std::string formatDurationFromString(const std::string& timestamp);
    std::string formatQuality(sep::train::Quality quality);
    std::string getStatusIcon(bool status);
    std::string getQualityIcon(sep::train::Quality quality);
    
    // Monitoring mode
    void monitoringLoop(int duration_seconds);
    void clearScreen();
    void printLiveMetrics();
};

}  // namespace sep

#endif  // STATUS_DISPLAY_HPP
