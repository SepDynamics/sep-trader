#pragma once

#include "cache_validator.hpp"
#include "weekly_cache_manager.hpp"
#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>
#include "engine/internal/standard_includes.h"
#include <atomic>
#include <thread>

namespace sep::cache {

// Cache health status levels
enum class HealthStatus {
    EXCELLENT,  // All caches healthy, no issues
    GOOD,       // Minor issues but functional
    WARNING,    // Some issues requiring attention
    CRITICAL,   // Major issues affecting functionality
    FAILURE     // System-wide cache failure
};

// Health metric categories
enum class HealthMetric {
    AVAILABILITY,   // Cache files exist and accessible
    FRESHNESS,      // Data recency and currency
    COMPLETENESS,   // Data coverage and gaps
    CONSISTENCY,    // Data integrity and validation
    PERFORMANCE,    // Access speed and efficiency
    RELIABILITY     // Error rates and stability
};

// Individual cache health report
struct CacheHealthReport {
    std::string pair_symbol;
    std::string cache_path;
    HealthStatus overall_status;
    std::unordered_map<HealthMetric, double> metric_scores; // 0.0-1.0
    std::unordered_map<HealthMetric, HealthStatus> metric_statuses;
    CacheQuality quality_metrics;
    WeeklyCacheStatus weekly_status;
    std::vector<std::string> issues;
    std::vector<std::string> recommendations;
    std::chrono::system_clock::time_point last_checked;
    std::chrono::duration<double> check_duration;
    
    CacheHealthReport() : overall_status(HealthStatus::FAILURE), 
                         last_checked(std::chrono::system_clock::now()),
                         check_duration(std::chrono::duration<double>::zero()) {}
};

// System-wide cache health summary
struct SystemCacheHealth {
    HealthStatus overall_status;
    size_t total_caches;
    size_t healthy_caches;
    size_t warning_caches;
    size_t critical_caches;
    size_t failed_caches;
    double average_health_score;
    double system_performance_score;
    std::vector<std::string> system_issues;
    std::vector<std::string> system_recommendations;
    std::chrono::system_clock::time_point last_assessment;
    
    SystemCacheHealth() : overall_status(HealthStatus::FAILURE), total_caches(0),
                         healthy_caches(0), warning_caches(0), critical_caches(0),
                         failed_caches(0), average_health_score(0.0),
                         system_performance_score(0.0),
                         last_assessment(std::chrono::system_clock::now()) {}
};

// Health monitoring configuration
struct MonitoringConfig {
    std::chrono::minutes check_interval{30}; // Check every 30 minutes
    std::chrono::minutes report_interval{240}; // Generate reports every 4 hours
    bool enable_automatic_repair{true}; // Attempt automatic repairs
    bool enable_predictive_alerts{true}; // Predict potential issues
    double warning_threshold{0.7}; // Health score threshold for warnings
    double critical_threshold{0.4}; // Health score threshold for critical
    size_t max_history_entries{1000}; // Maximum health history entries
    bool enable_performance_tracking{true}; // Track access performance
    
    MonitoringConfig() = default;
};

// Health event types
enum class HealthEventType {
    STATUS_CHANGE,      // Health status changed
    THRESHOLD_BREACH,   // Health score crossed threshold
    CACHE_FAILURE,      // Cache became inaccessible
    CACHE_RECOVERY,     // Cache recovered from failure
    PERFORMANCE_DEGRADATION, // Performance issues detected
    MAINTENANCE_REQUIRED,    // Maintenance action needed
    AUTOMATIC_REPAIR         // Automatic repair performed
};

// Health event
struct HealthEvent {
    HealthEventType type;
    std::string pair_symbol;
    std::string cache_path;
    HealthStatus old_status;
    HealthStatus new_status;
    double health_score;
    std::string description;
    std::chrono::system_clock::time_point timestamp;
    
    HealthEvent() : type(HealthEventType::STATUS_CHANGE), 
                   old_status(HealthStatus::FAILURE), 
                   new_status(HealthStatus::FAILURE),
                   health_score(0.0),
                   timestamp(std::chrono::system_clock::now()) {}
};

// Health monitoring callback types
using HealthEventCallback = std::function<void(const HealthEvent& event)>;
using HealthReportCallback = std::function<void(const SystemCacheHealth& system_health)>;
using MaintenanceCallback = std::function<bool(const std::string& pair_symbol, const std::vector<std::string>& issues)>;

class CacheHealthMonitor {
public:
    CacheHealthMonitor();
    ~CacheHealthMonitor();

    // Health monitoring control
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const;
    void setMonitoringConfig(const MonitoringConfig& config);
    MonitoringConfig getMonitoringConfig() const;
    
    // Health assessment
    CacheHealthReport assessCacheHealth(const std::string& pair_symbol);
    SystemCacheHealth assessSystemHealth();
    std::vector<CacheHealthReport> assessAllCaches();
    HealthStatus getOverallSystemStatus() const;
    
    // Individual cache monitoring
    void addCacheToMonitoring(const std::string& pair_symbol);
    void removeCacheFromMonitoring(const std::string& pair_symbol);
    std::vector<std::string> getMonitoredCaches() const;
    bool isCacheMonitored(const std::string& pair_symbol) const;
    
    // Health metrics
    double calculateHealthScore(const std::string& pair_symbol) const;
    double calculateMetricScore(const std::string& pair_symbol, HealthMetric metric) const;
    std::unordered_map<HealthMetric, double> getAllMetricScores(const std::string& pair_symbol) const;
    
    // Performance monitoring
    void recordCacheAccess(const std::string& pair_symbol, std::chrono::duration<double> access_time);
    void recordCacheError(const std::string& pair_symbol, const std::string& error_type);
    double getAverageAccessTime(const std::string& pair_symbol) const;
    double getErrorRate(const std::string& pair_symbol) const;
    
    // Alerting and notifications
    size_t addHealthEventCallback(HealthEventCallback callback);
    void removeHealthEventCallback(size_t callback_id);
    size_t addHealthReportCallback(HealthReportCallback callback);
    void removeHealthReportCallback(size_t callback_id);
    size_t addMaintenanceCallback(MaintenanceCallback callback);
    void removeMaintenanceCallback(size_t callback_id);
    
    // Health history and trending
    std::vector<HealthEvent> getHealthHistory(const std::string& pair_symbol = "") const;
    std::vector<double> getHealthTrend(const std::string& pair_symbol, std::chrono::hours period) const;
    bool isPredictiveAlertTriggered(const std::string& pair_symbol) const;
    std::vector<std::string> getPredictedIssues(const std::string& pair_symbol) const;
    
    // Maintenance and repair
    bool performAutomaticRepair(const std::string& pair_symbol);
    std::vector<std::string> getMaintenanceRecommendations(const std::string& pair_symbol) const;
    bool scheduleMaintenance(const std::string& pair_symbol, std::chrono::system_clock::time_point when);
    std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> getScheduledMaintenance() const;
    
    // Reporting and statistics
    SystemCacheHealth generateSystemReport();
    std::string generateDetailedReport(const std::string& pair_symbol = "") const;
    std::string generateSummaryReport() const;
    void exportHealthData(const std::string& output_path) const;
    
    // Configuration and thresholds
    void setHealthThresholds(double warning_threshold, double critical_threshold);
    std::pair<double, double> getHealthThresholds() const;
    void setCustomThresholds(const std::string& pair_symbol, double warning, double critical);
    bool hasCustomThresholds(const std::string& pair_symbol) const;
    
    // Integration with other systems
    void integrateCacheValidator(std::shared_ptr<CacheValidator> validator);
    void integrateWeeklyCacheManager(std::shared_ptr<WeeklyCacheManager> manager);
    bool hasValidatorIntegration() const;
    bool hasWeeklyCacheManagerIntegration() const;

private:
    mutable std::mutex monitor_mutex_;
    MonitoringConfig config_;
    std::atomic<bool> monitoring_active_{false};
    std::unique_ptr<std::thread> monitoring_thread_;
    std::atomic<bool> stop_monitoring_{false};
    
    // Cache tracking
    std::vector<std::string> monitored_caches_;
    std::unordered_map<std::string, CacheHealthReport> latest_reports_;
    SystemCacheHealth latest_system_health_;
    mutable std::mutex cache_data_mutex_;
    
    // Performance tracking
    struct PerformanceData {
        std::vector<std::chrono::duration<double>> access_times;
        std::vector<std::chrono::system_clock::time_point> error_times;
        std::vector<std::string> error_types;
        size_t total_accesses;
        
        PerformanceData() : total_accesses(0) {}
    };
    std::unordered_map<std::string, PerformanceData> performance_data_;
    mutable std::mutex performance_mutex_;
    
    // Health history
    std::vector<HealthEvent> health_history_;
    mutable std::mutex history_mutex_;
    
    // Custom thresholds
    std::unordered_map<std::string, std::pair<double, double>> custom_thresholds_;
    
    // Event callbacks
    std::vector<HealthEventCallback> health_event_callbacks_;
    std::vector<HealthReportCallback> health_report_callbacks_;
    std::vector<MaintenanceCallback> maintenance_callbacks_;
    mutable std::mutex callbacks_mutex_;
    
    // External integrations
    std::shared_ptr<CacheValidator> cache_validator_;
    std::shared_ptr<WeeklyCacheManager> weekly_cache_manager_;
    
    // Scheduled maintenance
    std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> scheduled_maintenance_;
    mutable std::mutex maintenance_mutex_;
    
    // Internal monitoring methods
    void monitoringLoop();
    void performHealthChecks();
    void generatePeriodicReports();
    void processMaintenanceSchedule();
    
    // Health calculation
    CacheHealthReport generateHealthReport(const std::string& pair_symbol);
    double calculateOverallScore(const std::unordered_map<HealthMetric, double>& metric_scores) const;
    HealthStatus scoreToStatus(double score) const;
    HealthStatus determineOverallStatus(const std::unordered_map<HealthMetric, HealthStatus>& statuses) const;
    
    // Metric calculations
    double calculateAvailabilityScore(const std::string& pair_symbol) const;
    double calculateFreshnessScore(const std::string& pair_symbol) const;
    double calculateCompletenessScore(const std::string& pair_symbol) const;
    double calculateConsistencyScore(const std::string& pair_symbol) const;
    double calculatePerformanceScore(const std::string& pair_symbol) const;
    double calculateReliabilityScore(const std::string& pair_symbol) const;
    
    // Issue detection and recommendations
    std::vector<std::string> detectIssues(const CacheHealthReport& report) const;
    std::vector<std::string> generateRecommendations(const CacheHealthReport& report) const;
    std::vector<std::string> predictFutureIssues(const std::string& pair_symbol) const;
    
    // Event handling
    void recordHealthEvent(const HealthEvent& event);
    void notifyHealthEvent(const HealthEvent& event);
    void notifyHealthReport(const SystemCacheHealth& system_health);
    bool requestMaintenance(const std::string& pair_symbol, const std::vector<std::string>& issues);
    
    // Utility methods
    std::pair<double, double> getThresholdsForPair(const std::string& pair_symbol) const;
    bool shouldTriggerAlert(double old_score, double new_score, const std::pair<double, double>& thresholds) const;
    std::string formatHealthReport(const CacheHealthReport& report) const;
    std::string formatSystemHealth(const SystemCacheHealth& system_health) const;
};

// Utility functions
std::string healthStatusToString(HealthStatus status);
HealthStatus stringToHealthStatus(const std::string& status_str);
std::string healthMetricToString(HealthMetric metric);
HealthMetric stringToHealthMetric(const std::string& metric_str);
std::string healthEventTypeToString(HealthEventType type);
HealthEventType stringToHealthEventType(const std::string& type_str);
bool isHealthStatusGood(HealthStatus status);

// Global cache health monitor instance
CacheHealthMonitor& getGlobalCacheHealthMonitor();

} // namespace sep::cache
