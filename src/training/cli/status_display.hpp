// SEP Training Status Display
// Professional status and monitoring display for training system

#ifndef STATUS_DISPLAY_HPP
#define STATUS_DISPLAY_HPP

#include <string>
#include <map>
#include <chrono>
#include "training/training_coordinator.hpp"

namespace sep {
namespace training {

class StatusDisplay {
public:
    explicit StatusDisplay(TrainingCoordinator& coordinator);
    
    // Status displays
    bool showSystemStatus();
    bool showSystemHealth();
    bool showTuningStatus();
    bool startMonitoringMode(int duration_seconds);
    
private:
    TrainingCoordinator& coordinator_;
    
    // Display utilities
    void printStatusHeader(const std::string& title);
    void printStatusLine(const std::string& label, const std::string& value, bool good = true);
    void printTrainingTable(const std::vector<TrainingResult>& results);
    void printRemoteStatus();
    void printCacheStatus();
    void printPerformanceMetrics();
    
    // Formatting utilities
    std::string formatAccuracy(double accuracy);
    std::string formatDuration(std::chrono::system_clock::time_point timestamp);
    std::string formatQuality(PatternQuality quality);
    std::string getStatusIcon(bool status);
    std::string getQualityIcon(PatternQuality quality);
    
    // Monitoring mode
    void monitoringLoop(int duration_seconds);
    void clearScreen();
    void printLiveMetrics();
};

} // namespace training
} // namespace sep

#endif // STATUS_DISPLAY_HPP
