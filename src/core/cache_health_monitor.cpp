#include "util/nlohmann_json_safe.h"
#include "cache_health_monitor.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>


namespace sep::cache {

namespace fs = std::filesystem;

CacheHealthMonitor::CacheHealthMonitor() {
    // Set default monitoring configuration
    config_.check_interval = std::chrono::minutes(30);
    config_.report_interval = std::chrono::minutes(240);
    config_.warning_threshold = 0.7;
    config_.critical_threshold = 0.4;
    
    spdlog::info("CacheHealthMonitor initialized");
}

CacheHealthMonitor::~CacheHealthMonitor() {
    stopMonitoring();
}

void CacheHealthMonitor::startMonitoring() {
    if (monitoring_active_) {
        spdlog::warn("Cache health monitoring is already active");
        return;
    }
    
    monitoring_active_ = true;
    stop_monitoring_ = false;
    monitoring_thread_ = std::make_unique<std::thread>(&CacheHealthMonitor::monitoringLoop, this);
    
    spdlog::info("Cache health monitoring started");
}

void CacheHealthMonitor::stopMonitoring() {
    if (!monitoring_active_) {
        return;
    }
    
    stop_monitoring_ = true;
    monitoring_active_ = false;
    
    if (monitoring_thread_ && monitoring_thread_->joinable()) {
        monitoring_thread_->join();
    }
    monitoring_thread_.reset();
    
    spdlog::info("Cache health monitoring stopped");
}

bool CacheHealthMonitor::isMonitoring() const {
    return monitoring_active_;
}

void CacheHealthMonitor::setMonitoringConfig(const MonitoringConfig& config) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    config_ = config;
    spdlog::info("Cache health monitoring configuration updated");
}

MonitoringConfig CacheHealthMonitor::getMonitoringConfig() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return config_;
}

CacheHealthReport CacheHealthMonitor::assessCacheHealth(const std::string& pair_symbol) {
    std::lock_guard<std::mutex> lock(cache_data_mutex_);
    
    auto report = generateHealthReport(pair_symbol);
    latest_reports_[pair_symbol] = report;
    
    // Record health event if status changed
    auto history_it = std::find_if(health_history_.rbegin(), health_history_.rend(),
                                  [&pair_symbol](const HealthEvent& event) {
                                      return event.pair_symbol == pair_symbol;
                                  });
    
    HealthStatus old_status = (history_it != health_history_.rend()) ? 
                             history_it->new_status : HealthStatus::FAILURE;
    
    if (report.overall_status != old_status) {
        HealthEvent event;
        event.type = HealthEventType::STATUS_CHANGE;
        event.pair_symbol = pair_symbol;
        event.old_status = old_status;
        event.new_status = report.overall_status;
        event.health_score = calculateHealthScore(pair_symbol);
        event.description = "Health status changed from " + healthStatusToString(old_status) + 
                           " to " + healthStatusToString(report.overall_status);
        
        recordHealthEvent(event);
    }
    
    return report;
}

SystemCacheHealth CacheHealthMonitor::assessSystemHealth() {
    std::lock_guard<std::mutex> lock(cache_data_mutex_);
    
    SystemCacheHealth system_health;
    system_health.last_assessment = std::chrono::system_clock::now();
    
    // Assess all monitored caches
    std::vector<double> health_scores;
    
    for (const auto& pair_symbol : monitored_caches_) {
        auto report = generateHealthReport(pair_symbol);
        latest_reports_[pair_symbol] = report;
        
        system_health.total_caches++;
        
        switch (report.overall_status) {
            case HealthStatus::EXCELLENT:
            case HealthStatus::GOOD:
                system_health.healthy_caches++;
                break;
            case HealthStatus::WARNING:
                system_health.warning_caches++;
                break;
            case HealthStatus::CRITICAL:
                system_health.critical_caches++;
                break;
            case HealthStatus::FAILURE:
                system_health.failed_caches++;
                break;
        }
        
        double score = calculateHealthScore(pair_symbol);
        health_scores.push_back(score);
    }
    
    // Calculate system-wide metrics
    if (!health_scores.empty()) {
        system_health.average_health_score = std::accumulate(health_scores.begin(), health_scores.end(), 0.0) / health_scores.size();
        
        // Calculate performance score based on healthy vs total caches
        system_health.system_performance_score = 
            static_cast<double>(system_health.healthy_caches) / system_health.total_caches;
    }
    
    // Determine overall system status
    if (system_health.failed_caches > system_health.total_caches * 0.5) {
        system_health.overall_status = HealthStatus::FAILURE;
        system_health.system_issues.push_back("More than 50% of caches have failed");
    } else if (system_health.critical_caches > system_health.total_caches * 0.3) {
        system_health.overall_status = HealthStatus::CRITICAL;
        system_health.system_issues.push_back("More than 30% of caches are in critical state");
    } else if (system_health.warning_caches > system_health.total_caches * 0.2) {
        system_health.overall_status = HealthStatus::WARNING;
        system_health.system_issues.push_back("More than 20% of caches have warnings");
    } else if (system_health.average_health_score > 0.8) {
        system_health.overall_status = HealthStatus::EXCELLENT;
    } else {
        system_health.overall_status = HealthStatus::GOOD;
    }
    
    // Generate system recommendations
    if (system_health.failed_caches > 0) {
        system_health.system_recommendations.push_back("Rebuild failed caches immediately");
    }
    if (system_health.critical_caches > 0) {
        system_health.system_recommendations.push_back("Investigate critical cache issues");
    }
    if (system_health.average_health_score < 0.6) {
        system_health.system_recommendations.push_back("Perform system-wide cache maintenance");
    }
    
    latest_system_health_ = system_health;
    return system_health;
}

std::vector<CacheHealthReport> CacheHealthMonitor::assessAllCaches() {
    std::vector<CacheHealthReport> reports;
    
    for (const auto& pair_symbol : monitored_caches_) {
        reports.push_back(assessCacheHealth(pair_symbol));
    }
    
    return reports;
}

HealthStatus CacheHealthMonitor::getOverallSystemStatus() const {
    std::lock_guard<std::mutex> lock(cache_data_mutex_);
    return latest_system_health_.overall_status;
}

void CacheHealthMonitor::addCacheToMonitoring(const std::string& pair_symbol) {
    std::lock_guard<std::mutex> lock(cache_data_mutex_);
    
    auto it = std::find(monitored_caches_.begin(), monitored_caches_.end(), pair_symbol);
    if (it == monitored_caches_.end()) {
        monitored_caches_.push_back(pair_symbol);
        spdlog::info("Added {} to cache health monitoring", pair_symbol);
    }
}

void CacheHealthMonitor::removeCacheFromMonitoring(const std::string& pair_symbol) {
    std::lock_guard<std::mutex> lock(cache_data_mutex_);
    
    auto it = std::find(monitored_caches_.begin(), monitored_caches_.end(), pair_symbol);
    if (it != monitored_caches_.end()) {
        monitored_caches_.erase(it);
        latest_reports_.erase(pair_symbol);
        spdlog::info("Removed {} from cache health monitoring", pair_symbol);
    }
}

std::vector<std::string> CacheHealthMonitor::getMonitoredCaches() const {
    std::lock_guard<std::mutex> lock(cache_data_mutex_);
    return monitored_caches_;
}

bool CacheHealthMonitor::isCacheMonitored(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(cache_data_mutex_);
    return std::find(monitored_caches_.begin(), monitored_caches_.end(), pair_symbol) != monitored_caches_.end();
}

double CacheHealthMonitor::calculateHealthScore(const std::string& pair_symbol) const {
    auto metric_scores = getAllMetricScores(pair_symbol);
    
    if (metric_scores.empty()) return 0.0;
    
    // Weighted average of all metrics
    double total_score = 0.0;
    double total_weight = 0.0;
    
    // Define weights for different metrics
    std::unordered_map<HealthMetric, double> weights = {
        {HealthMetric::AVAILABILITY, 0.25},
        {HealthMetric::FRESHNESS, 0.20},
        {HealthMetric::COMPLETENESS, 0.20},
        {HealthMetric::CONSISTENCY, 0.15},
        {HealthMetric::PERFORMANCE, 0.10},
        {HealthMetric::RELIABILITY, 0.10}
    };
    
    for (const auto& [metric, score] : metric_scores) {
        double weight = weights[metric];
        total_score += score * weight;
        total_weight += weight;
    }
    
    return (total_weight > 0) ? total_score / total_weight : 0.0;
}

double CacheHealthMonitor::calculateMetricScore(const std::string& pair_symbol, HealthMetric metric) const {
    switch (metric) {
        case HealthMetric::AVAILABILITY:
            return calculateAvailabilityScore(pair_symbol);
        case HealthMetric::FRESHNESS:
            return calculateFreshnessScore(pair_symbol);
        case HealthMetric::COMPLETENESS:
            return calculateCompletenessScore(pair_symbol);
        case HealthMetric::CONSISTENCY:
            return calculateConsistencyScore(pair_symbol);
        case HealthMetric::PERFORMANCE:
            return calculatePerformanceScore(pair_symbol);
        case HealthMetric::RELIABILITY:
            return calculateReliabilityScore(pair_symbol);
        default:
            return 0.0;
    }
}

std::unordered_map<HealthMetric, double> CacheHealthMonitor::getAllMetricScores(const std::string& pair_symbol) const {
    std::unordered_map<HealthMetric, double> scores;
    
    scores[HealthMetric::AVAILABILITY] = calculateAvailabilityScore(pair_symbol);
    scores[HealthMetric::FRESHNESS] = calculateFreshnessScore(pair_symbol);
    scores[HealthMetric::COMPLETENESS] = calculateCompletenessScore(pair_symbol);
    scores[HealthMetric::CONSISTENCY] = calculateConsistencyScore(pair_symbol);
    scores[HealthMetric::PERFORMANCE] = calculatePerformanceScore(pair_symbol);
    scores[HealthMetric::RELIABILITY] = calculateReliabilityScore(pair_symbol);
    
    return scores;
}

void CacheHealthMonitor::recordCacheAccess(const std::string& pair_symbol, std::chrono::duration<double> access_time) {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    auto& perf_data = performance_data_[pair_symbol];
    perf_data.access_times.push_back(access_time);
    perf_data.total_accesses++;
    
    // Keep only recent access times (last 1000 entries)
    if (perf_data.access_times.size() > 1000) {
        perf_data.access_times.erase(perf_data.access_times.begin());
    }
}

void CacheHealthMonitor::recordCacheError(const std::string& pair_symbol, const std::string& error_type) {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    auto& perf_data = performance_data_[pair_symbol];
    perf_data.error_times.push_back(std::chrono::system_clock::now());
    perf_data.error_types.push_back(error_type);
    
    // Keep only recent errors (last 500 entries)
    if (perf_data.error_times.size() > 500) {
        perf_data.error_times.erase(perf_data.error_times.begin());
        perf_data.error_types.erase(perf_data.error_types.begin());
    }
    
    spdlog::warn("Cache error recorded for {}: {}", pair_symbol, error_type);
}

double CacheHealthMonitor::getAverageAccessTime(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    auto it = performance_data_.find(pair_symbol);
    if (it == performance_data_.end() || it->second.access_times.empty()) {
        return 0.0;
    }
    
    const auto& access_times = it->second.access_times;
    double total_time = std::accumulate(access_times.begin(), access_times.end(), 
                                       std::chrono::duration<double>::zero()).count();
    
    return total_time / access_times.size();
}

double CacheHealthMonitor::getErrorRate(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    auto it = performance_data_.find(pair_symbol);
    if (it == performance_data_.end()) {
        return 0.0;
    }
    
    const auto& perf_data = it->second;
    if (perf_data.total_accesses == 0) {
        return 0.0;
    }
    
    return static_cast<double>(perf_data.error_times.size()) / perf_data.total_accesses;
}

size_t CacheHealthMonitor::addHealthEventCallback(HealthEventCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    health_event_callbacks_.push_back(std::move(callback));
    return health_event_callbacks_.size() - 1;
}

void CacheHealthMonitor::removeHealthEventCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < health_event_callbacks_.size()) {
        health_event_callbacks_.erase(health_event_callbacks_.begin() + callback_id);
    }
}

size_t CacheHealthMonitor::addHealthReportCallback(HealthReportCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    health_report_callbacks_.push_back(std::move(callback));
    return health_report_callbacks_.size() - 1;
}

void CacheHealthMonitor::removeHealthReportCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < health_report_callbacks_.size()) {
        health_report_callbacks_.erase(health_report_callbacks_.begin() + callback_id);
    }
}

size_t CacheHealthMonitor::addMaintenanceCallback(MaintenanceCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    maintenance_callbacks_.push_back(std::move(callback));
    return maintenance_callbacks_.size() - 1;
}

void CacheHealthMonitor::removeMaintenanceCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < maintenance_callbacks_.size()) {
        maintenance_callbacks_.erase(maintenance_callbacks_.begin() + callback_id);
    }
}

std::vector<HealthEvent> CacheHealthMonitor::getHealthHistory(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    if (pair_symbol.empty()) {
        return health_history_;
    }
    
    std::vector<HealthEvent> filtered_history;
    std::copy_if(health_history_.begin(), health_history_.end(), 
                std::back_inserter(filtered_history),
                [&pair_symbol](const HealthEvent& event) {
                    return event.pair_symbol == pair_symbol;
                });
    
    return filtered_history;
}

std::vector<double> CacheHealthMonitor::getHealthTrend(const std::string& pair_symbol, std::chrono::hours period) const {
    std::vector<double> trend;
    auto history = getHealthHistory(pair_symbol);
    
    auto cutoff_time = std::chrono::system_clock::now() - period;
    
    for (const auto& event : history) {
        if (event.timestamp >= cutoff_time && event.type == HealthEventType::STATUS_CHANGE) {
            trend.push_back(event.health_score);
        }
    }
    
    return trend;
}

bool CacheHealthMonitor::isPredictiveAlertTriggered(const std::string& pair_symbol) const {
    auto trend = getHealthTrend(pair_symbol, std::chrono::hours(24));
    
    if (trend.size() < 3) return false;
    
    // Simple predictive logic: if health score is consistently declining
    bool declining = true;
    for (size_t i = 1; i < trend.size(); ++i) {
        if (trend[i] >= trend[i-1]) {
            declining = false;
            break;
        }
    }
    
    return declining && trend.back() < config_.warning_threshold;
}

std::vector<std::string> CacheHealthMonitor::getPredictedIssues(const std::string& pair_symbol) const {
    return predictFutureIssues(pair_symbol);
}

bool CacheHealthMonitor::performAutomaticRepair(const std::string& pair_symbol) {
    if (!config_.enable_automatic_repair) {
        return false;
    }
    
    auto it = latest_reports_.find(pair_symbol);
    if (it == latest_reports_.end()) {
        return false;
    }
    
    const auto& report = it->second;
    return requestMaintenance(pair_symbol, report.issues);
}

std::vector<std::string> CacheHealthMonitor::getMaintenanceRecommendations(const std::string& pair_symbol) const {
    auto it = latest_reports_.find(pair_symbol);
    if (it != latest_reports_.end()) {
        return generateRecommendations(it->second);
    }
    return {};
}

bool CacheHealthMonitor::scheduleMaintenance(const std::string& pair_symbol, std::chrono::system_clock::time_point when) {
    std::lock_guard<std::mutex> lock(maintenance_mutex_);
    
    scheduled_maintenance_.emplace_back(pair_symbol, when);
    spdlog::info("Scheduled maintenance for {} at {}", pair_symbol, 
                std::chrono::duration_cast<std::chrono::seconds>(when.time_since_epoch()).count());
    
    return true;
}

std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> CacheHealthMonitor::getScheduledMaintenance() const {
    std::lock_guard<std::mutex> lock(maintenance_mutex_);
    return scheduled_maintenance_;
}

SystemCacheHealth CacheHealthMonitor::generateSystemReport() {
    return assessSystemHealth();
}

std::string CacheHealthMonitor::generateDetailedReport(const std::string& pair_symbol) const {
    std::ostringstream report;
    
    if (pair_symbol.empty()) {
        // System-wide report
        report << "=== CACHE HEALTH SYSTEM REPORT ===\n";
        report << formatSystemHealth(latest_system_health_);
        
        report << "\n=== INDIVIDUAL CACHE REPORTS ===\n";
        for (const auto& [pair, cache_report] : latest_reports_) {
            report << formatHealthReport(cache_report) << "\n";
        }
    } else {
        // Individual cache report
        auto it = latest_reports_.find(pair_symbol);
        if (it != latest_reports_.end()) {
            report << formatHealthReport(it->second);
        } else {
            report << "No health report available for pair: " << pair_symbol;
        }
    }
    
    return report.str();
}

std::string CacheHealthMonitor::generateSummaryReport() const {
    std::ostringstream report;
    
    report << "Cache Health Summary:\n";
    report << "  Total Caches: " << latest_system_health_.total_caches << "\n";
    report << "  Healthy: " << latest_system_health_.healthy_caches << "\n";
    report << "  Warning: " << latest_system_health_.warning_caches << "\n";
    report << "  Critical: " << latest_system_health_.critical_caches << "\n";
    report << "  Failed: " << latest_system_health_.failed_caches << "\n";
    report << "  Overall Status: " << healthStatusToString(latest_system_health_.overall_status) << "\n";
    report << "  Average Health Score: " << std::fixed << std::setprecision(2) 
           << latest_system_health_.average_health_score << "\n";
    
    return report.str();
}

void CacheHealthMonitor::exportHealthData(const std::string& output_path) const {
    try {
        nlohmann::json root;
        nlohmann::json system_health;
        
        // Export system health
        system_health["overall_status"] = healthStatusToString(latest_system_health_.overall_status);
        system_health["total_caches"] = static_cast<int>(latest_system_health_.total_caches);
        system_health["healthy_caches"] = static_cast<int>(latest_system_health_.healthy_caches);
        system_health["warning_caches"] = static_cast<int>(latest_system_health_.warning_caches);
        system_health["critical_caches"] = static_cast<int>(latest_system_health_.critical_caches);
        system_health["failed_caches"] = static_cast<int>(latest_system_health_.failed_caches);
        system_health["average_health_score"] = latest_system_health_.average_health_score;
        system_health["system_performance_score"] = latest_system_health_.system_performance_score;
        
        root["system_health"] = system_health;
        
        // Export individual cache reports
        nlohmann::json cache_reports = nlohmann::json::array();
        for (const auto& [pair, report] : latest_reports_) {
            nlohmann::json cache_report;
            cache_report["pair_symbol"] = pair;
            cache_report["overall_status"] = healthStatusToString(report.overall_status);
            cache_report["cache_path"] = report.cache_path;
            cache_report["check_duration"] = report.check_duration.count();
            
            nlohmann::json metric_scores;
            for (const auto& [metric, score] : report.metric_scores) {
                metric_scores[healthMetricToString(metric)] = score;
            }
            cache_report["metric_scores"] = metric_scores;
            
            cache_reports.push_back(cache_report);
        }
        root["cache_reports"] = cache_reports;
        
        // Export health history
        nlohmann::json history = nlohmann::json::array();
        for (const auto& event : health_history_) {
            nlohmann::json event_json;
            event_json["type"] = healthEventTypeToString(event.type);
            event_json["pair_symbol"] = event.pair_symbol;
            event_json["old_status"] = healthStatusToString(event.old_status);
            event_json["new_status"] = healthStatusToString(event.new_status);
            event_json["health_score"] = event.health_score;
            event_json["description"] = event.description;
            event_json["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
                event.timestamp.time_since_epoch()).count();
            
            history.push_back(event_json);
        }
        root["health_history"] = history;
        
        // Write to file
        std::ofstream file(output_path);
        file << root.dump(2);  // Pretty print with 2-space indentation
        
        spdlog::info("Health data exported to: {}", output_path);
        
    } catch (const std::exception& e) {
        spdlog::error("Failed to export health data to {}: {}", output_path, e.what());
    }
}

void CacheHealthMonitor::setHealthThresholds(double warning_threshold, double critical_threshold) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    config_.warning_threshold = warning_threshold;
    config_.critical_threshold = critical_threshold;
    spdlog::info("Updated health thresholds: warning={:.2f}, critical={:.2f}", 
                warning_threshold, critical_threshold);
}

std::pair<double, double> CacheHealthMonitor::getHealthThresholds() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return {config_.warning_threshold, config_.critical_threshold};
}

void CacheHealthMonitor::setCustomThresholds(const std::string& pair_symbol, double warning, double critical) {
    custom_thresholds_[pair_symbol] = {warning, critical};
    spdlog::info("Set custom thresholds for {}: warning={:.2f}, critical={:.2f}", 
                pair_symbol, warning, critical);
}

bool CacheHealthMonitor::hasCustomThresholds(const std::string& pair_symbol) const {
    return custom_thresholds_.find(pair_symbol) != custom_thresholds_.end();
}

void CacheHealthMonitor::integrateCacheValidator(std::shared_ptr<CacheValidator> validator) {
    cache_validator_ = validator;
    spdlog::info("Integrated cache validator");
}

void CacheHealthMonitor::integrateWeeklyCacheManager(std::shared_ptr<WeeklyCacheManager> manager) {
    weekly_cache_manager_ = manager;
    spdlog::info("Integrated weekly cache manager");
}

bool CacheHealthMonitor::hasValidatorIntegration() const {
    return static_cast<bool>(cache_validator_);
}

bool CacheHealthMonitor::hasWeeklyCacheManagerIntegration() const {
    return static_cast<bool>(weekly_cache_manager_);
}

// Private method implementations

void CacheHealthMonitor::monitoringLoop() {
    auto last_check = std::chrono::steady_clock::now();
    auto last_report = std::chrono::steady_clock::now();
    
    while (!stop_monitoring_) {
        auto now = std::chrono::steady_clock::now();
        
        // Perform health checks
        if (now - last_check >= config_.check_interval) {
            performHealthChecks();
            last_check = now;
        }
        
        // Generate periodic reports
        if (now - last_report >= config_.report_interval) {
            generatePeriodicReports();
            last_report = now;
        }
        
        // Process maintenance schedule
        processMaintenanceSchedule();
        
        // Sleep for a short interval
        std::this_thread::sleep_for(std::chrono::seconds(30));
    }
}

void CacheHealthMonitor::performHealthChecks() {
    for (const auto& pair_symbol : monitored_caches_) {
        assessCacheHealth(pair_symbol);
    }
}

void CacheHealthMonitor::generatePeriodicReports() {
    auto system_health = assessSystemHealth();
    notifyHealthReport(system_health);
    
    spdlog::info("Periodic health report generated - Overall status: {}", 
                healthStatusToString(system_health.overall_status));
}

void CacheHealthMonitor::processMaintenanceSchedule() {
    std::lock_guard<std::mutex> lock(maintenance_mutex_);
    
    auto now = std::chrono::system_clock::now();
    auto it = scheduled_maintenance_.begin();
    
    while (it != scheduled_maintenance_.end()) {
        if (it->second <= now) {
            const std::string& pair_symbol = it->first;
            
            // Trigger maintenance
            performAutomaticRepair(pair_symbol);
            
            spdlog::info("Scheduled maintenance executed for: {}", pair_symbol);
            it = scheduled_maintenance_.erase(it);
        } else {
            ++it;
        }
    }
}

CacheHealthReport CacheHealthMonitor::generateHealthReport(const std::string& pair_symbol) {
    auto start_time = std::chrono::steady_clock::now();
    
    CacheHealthReport report;
    report.pair_symbol = pair_symbol;
    report.last_checked = std::chrono::system_clock::now();
    
    // Get metric scores
    report.metric_scores = getAllMetricScores(pair_symbol);
    
    // Calculate metric statuses
    for (const auto& [metric, score] : report.metric_scores) {
        report.metric_statuses[metric] = scoreToStatus(score);
    }
    
    // Calculate overall score and status
    double overall_score = calculateOverallScore(report.metric_scores);
    report.overall_status = scoreToStatus(overall_score);
    
    // Integrate with cache validator if available
    if (cache_validator_) {
        std::string cache_path = cache_validator_->getCachePathForPair(pair_symbol);
        report.cache_path = cache_path;
        report.quality_metrics = cache_validator_->analyzeCacheQuality(cache_path);
    }
    
    // Integrate with weekly cache manager if available
    if (weekly_cache_manager_) {
        report.weekly_status = weekly_cache_manager_->checkWeeklyCacheStatus(pair_symbol);
    }
    
    // Detect issues and generate recommendations
    report.issues = detectIssues(report);
    report.recommendations = generateRecommendations(report);
    
    auto end_time = std::chrono::steady_clock::now();
    report.check_duration = std::chrono::duration<double>(end_time - start_time);
    
    return report;
}

double CacheHealthMonitor::calculateOverallScore(const std::unordered_map<HealthMetric, double>& metric_scores) const {
    if (metric_scores.empty()) return 0.0;
    
    double total_score = std::accumulate(metric_scores.begin(), metric_scores.end(), 0.0,
                                        [](double sum, const auto& pair) {
                                            return sum + pair.second;
                                        });
    
    return total_score / metric_scores.size();
}

HealthStatus CacheHealthMonitor::scoreToStatus(double score) const {
    if (score >= 0.9) return HealthStatus::EXCELLENT;
    if (score >= config_.warning_threshold) return HealthStatus::GOOD;
    if (score >= config_.critical_threshold) return HealthStatus::WARNING;
    if (score > 0.0) return HealthStatus::CRITICAL;
    return HealthStatus::FAILURE;
}

HealthStatus CacheHealthMonitor::determineOverallStatus(const std::unordered_map<HealthMetric, HealthStatus>& statuses) const {
    if (statuses.empty()) return HealthStatus::FAILURE;
    
    // Overall status is the worst individual status
    HealthStatus worst = HealthStatus::EXCELLENT;
    for (const auto& [metric, status] : statuses) {
        if (status < worst) {
            worst = status;
        }
    }
    
    return worst;
}

double CacheHealthMonitor::calculateAvailabilityScore(const std::string& pair_symbol) const {
    if (cache_validator_) {
        return cache_validator_->cacheExistsForPair(pair_symbol) ? 1.0 : 0.0;
    }
    return 0.0;
}

double CacheHealthMonitor::calculateFreshnessScore(const std::string& pair_symbol) const {
    if (cache_validator_) {
        auto quality = cache_validator_->analyzeCacheQuality(
            cache_validator_->getCachePathForPair(pair_symbol));
        return quality.data_freshness;
    }
    return 0.0;
}

double CacheHealthMonitor::calculateCompletenessScore(const std::string& pair_symbol) const {
    if (cache_validator_) {
        auto quality = cache_validator_->analyzeCacheQuality(
            cache_validator_->getCachePathForPair(pair_symbol));
        return quality.data_completeness;
    }
    return 0.0;
}

double CacheHealthMonitor::calculateConsistencyScore(const std::string& pair_symbol) const {
    if (cache_validator_) {
        auto quality = cache_validator_->analyzeCacheQuality(
            cache_validator_->getCachePathForPair(pair_symbol));
        return quality.data_consistency;
    }
    return 0.0;
}

double CacheHealthMonitor::calculatePerformanceScore(const std::string& pair_symbol) const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    auto it = performance_data_.find(pair_symbol);
    if (it == performance_data_.end() || it->second.access_times.empty()) {
        return 1.0; // No performance data means perfect performance
    }
    
    double avg_time = getAverageAccessTime(pair_symbol);
    
    // Score based on access time (faster = better)
    // Assuming 1 second is excellent, 5 seconds is poor
    if (avg_time <= 1.0) return 1.0;
    if (avg_time >= 5.0) return 0.0;
    return 1.0 - ((avg_time - 1.0) / 4.0);
}

double CacheHealthMonitor::calculateReliabilityScore(const std::string& pair_symbol) const {
    double error_rate = getErrorRate(pair_symbol);
    
    // Score based on error rate (lower = better)
    if (error_rate <= 0.01) return 1.0; // 1% or less errors is excellent
    if (error_rate >= 0.20) return 0.0; // 20% or more errors is failure
    return 1.0 - ((error_rate - 0.01) / 0.19);
}

std::vector<std::string> CacheHealthMonitor::detectIssues(const CacheHealthReport& report) const {
    std::vector<std::string> issues;
    
    for (const auto& [metric, status] : report.metric_statuses) {
        if (status == HealthStatus::CRITICAL || status == HealthStatus::FAILURE) {
            issues.push_back("Critical issue with " + healthMetricToString(metric));
        } else if (status == HealthStatus::WARNING) {
            issues.push_back("Warning for " + healthMetricToString(metric));
        }
    }
    
    if (report.weekly_status == WeeklyCacheStatus::STALE) {
        issues.push_back("Weekly cache is stale");
    } else if (report.weekly_status == WeeklyCacheStatus::MISSING) {
        issues.push_back("Weekly cache is missing");
    }
    
    return issues;
}

std::vector<std::string> CacheHealthMonitor::generateRecommendations(const CacheHealthReport& report) const {
    std::vector<std::string> recommendations;
    
    if (report.overall_status == HealthStatus::FAILURE) {
        recommendations.push_back("Rebuild cache from source");
        recommendations.push_back("Check data source connectivity");
    } else if (report.overall_status == HealthStatus::CRITICAL) {
        recommendations.push_back("Perform immediate cache maintenance");
        recommendations.push_back("Investigate data quality issues");
    } else if (report.overall_status == HealthStatus::WARNING) {
        recommendations.push_back("Schedule routine maintenance");
        recommendations.push_back("Monitor cache performance closely");
    }
    
    if (report.weekly_status != WeeklyCacheStatus::CURRENT) {
        recommendations.push_back("Update weekly cache data");
    }
    
    for (const auto& [metric, status] : report.metric_statuses) {
        if (status == HealthStatus::CRITICAL || status == HealthStatus::FAILURE) {
            if (metric == HealthMetric::PERFORMANCE) {
                recommendations.push_back("Optimize cache access patterns");
            } else if (metric == HealthMetric::RELIABILITY) {
                recommendations.push_back("Investigate and fix error sources");
            }
        }
    }
    
    return recommendations;
}

std::vector<std::string> CacheHealthMonitor::predictFutureIssues(const std::string& pair_symbol) const {
    std::vector<std::string> predictions;
    
    if (!config_.enable_predictive_alerts) {
        return predictions;
    }
    
    auto trend = getHealthTrend(pair_symbol, std::chrono::hours(24));
    
    if (trend.size() >= 3) {
        bool declining = true;
        for (size_t i = 1; i < trend.size(); ++i) {
            if (trend[i] >= trend[i-1]) {
                declining = false;
                break;
            }
        }
        
        if (declining) {
            predictions.push_back("Health score is trending downward");
            
            if (trend.back() < config_.critical_threshold * 1.2) {
                predictions.push_back("Cache may become critical soon");
            }
        }
    }
    
    double error_rate = getErrorRate(pair_symbol);
    if (error_rate > 0.05) {
        predictions.push_back("High error rate may lead to reliability issues");
    }
    
    return predictions;
}

void CacheHealthMonitor::recordHealthEvent(const HealthEvent& event) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    health_history_.push_back(event);
    
    // Keep only recent history
    if (health_history_.size() > config_.max_history_entries) {
        health_history_.erase(health_history_.begin());
    }
    
    notifyHealthEvent(event);
}

void CacheHealthMonitor::notifyHealthEvent(const HealthEvent& event) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : health_event_callbacks_) {
        try {
            callback(event);
        } catch (const std::exception& e) {
            spdlog::error("Health event callback error: {}", e.what());
        }
    }
}

void CacheHealthMonitor::notifyHealthReport(const SystemCacheHealth& system_health) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : health_report_callbacks_) {
        try {
            callback(system_health);
        } catch (const std::exception& e) {
            spdlog::error("Health report callback error: {}", e.what());
        }
    }
}

bool CacheHealthMonitor::requestMaintenance(const std::string& pair_symbol, const std::vector<std::string>& issues) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : maintenance_callbacks_) {
        try {
            if (callback(pair_symbol, issues)) {
                return true;
            }
        } catch (const std::exception& e) {
            spdlog::error("Maintenance callback error: {}", e.what());
        }
    }
    
    return false;
}

std::pair<double, double> CacheHealthMonitor::getThresholdsForPair(const std::string& pair_symbol) const {
    auto it = custom_thresholds_.find(pair_symbol);
    if (it != custom_thresholds_.end()) {
        return it->second;
    }
    return {config_.warning_threshold, config_.critical_threshold};
}

bool CacheHealthMonitor::shouldTriggerAlert(double old_score, double new_score, const std::pair<double, double>& thresholds) const {
    double warning_threshold = thresholds.first;
    double critical_threshold = thresholds.second;
    
    // Trigger alert if crossing thresholds downward
    return (old_score >= warning_threshold && new_score < warning_threshold) ||
           (old_score >= critical_threshold && new_score < critical_threshold);
}

std::string CacheHealthMonitor::formatHealthReport(const CacheHealthReport& report) const {
    std::ostringstream fmt;
    
    fmt << "=== Cache Health Report: " << report.pair_symbol << " ===\n";
    fmt << "Overall Status: " << healthStatusToString(report.overall_status) << "\n";
    fmt << "Cache Path: " << report.cache_path << "\n";
    fmt << "Last Checked: " << std::chrono::duration_cast<std::chrono::seconds>(
        report.last_checked.time_since_epoch()).count() << "\n";
    fmt << "Check Duration: " << std::fixed << std::setprecision(3) 
        << report.check_duration.count() * 1000 << "ms\n";
    
    fmt << "\nMetric Scores:\n";
    for (const auto& [metric, score] : report.metric_scores) {
        fmt << "  " << healthMetricToString(metric) << ": " 
            << std::fixed << std::setprecision(2) << score << " ("
            << healthStatusToString(report.metric_statuses.at(metric)) << ")\n";
    }
    
    if (!report.issues.empty()) {
        fmt << "\nIssues:\n";
        for (const auto& issue : report.issues) {
            fmt << "  - " << issue << "\n";
        }
    }
    
    if (!report.recommendations.empty()) {
        fmt << "\nRecommendations:\n";
        for (const auto& rec : report.recommendations) {
            fmt << "  - " << rec << "\n";
        }
    }
    
    return fmt.str();
}

std::string CacheHealthMonitor::formatSystemHealth(const SystemCacheHealth& system_health) const {
    std::ostringstream fmt;
    
    fmt << "Overall Status: " << healthStatusToString(system_health.overall_status) << "\n";
    fmt << "Total Caches: " << system_health.total_caches << "\n";
    fmt << "Healthy: " << system_health.healthy_caches << "\n";
    fmt << "Warning: " << system_health.warning_caches << "\n";
    fmt << "Critical: " << system_health.critical_caches << "\n";
    fmt << "Failed: " << system_health.failed_caches << "\n";
    fmt << "Average Health Score: " << std::fixed << std::setprecision(2) 
        << system_health.average_health_score << "\n";
    fmt << "System Performance Score: " << std::fixed << std::setprecision(2) 
        << system_health.system_performance_score << "\n";
    
    if (!system_health.system_issues.empty()) {
        fmt << "\nSystem Issues:\n";
        for (const auto& issue : system_health.system_issues) {
            fmt << "  - " << issue << "\n";
        }
    }
    
    if (!system_health.system_recommendations.empty()) {
        fmt << "\nSystem Recommendations:\n";
        for (const auto& rec : system_health.system_recommendations) {
            fmt << "  - " << rec << "\n";
        }
    }
    
    return fmt.str();
}

// Utility functions implementation

std::string healthStatusToString(HealthStatus status) {
    switch (status) {
        case HealthStatus::EXCELLENT: return "EXCELLENT";
        case HealthStatus::GOOD: return "GOOD";
        case HealthStatus::WARNING: return "WARNING";
        case HealthStatus::CRITICAL: return "CRITICAL";
        case HealthStatus::FAILURE: return "FAILURE";
        default: return "UNKNOWN";
    }
}

HealthStatus stringToHealthStatus(const std::string& status_str) {
    if (status_str == "EXCELLENT") return HealthStatus::EXCELLENT;
    if (status_str == "GOOD") return HealthStatus::GOOD;
    if (status_str == "WARNING") return HealthStatus::WARNING;
    if (status_str == "CRITICAL") return HealthStatus::CRITICAL;
    if (status_str == "FAILURE") return HealthStatus::FAILURE;
    return HealthStatus::FAILURE; // Default for unknown
}

std::string healthMetricToString(HealthMetric metric) {
    switch (metric) {
        case HealthMetric::AVAILABILITY: return "AVAILABILITY";
        case HealthMetric::FRESHNESS: return "FRESHNESS";
        case HealthMetric::COMPLETENESS: return "COMPLETENESS";
        case HealthMetric::CONSISTENCY: return "CONSISTENCY";
        case HealthMetric::PERFORMANCE: return "PERFORMANCE";
        case HealthMetric::RELIABILITY: return "RELIABILITY";
        default: return "UNKNOWN";
    }
}

HealthMetric stringToHealthMetric(const std::string& metric_str) {
    if (metric_str == "AVAILABILITY") return HealthMetric::AVAILABILITY;
    if (metric_str == "FRESHNESS") return HealthMetric::FRESHNESS;
    if (metric_str == "COMPLETENESS") return HealthMetric::COMPLETENESS;
    if (metric_str == "CONSISTENCY") return HealthMetric::CONSISTENCY;
    if (metric_str == "PERFORMANCE") return HealthMetric::PERFORMANCE;
    if (metric_str == "RELIABILITY") return HealthMetric::RELIABILITY;
    return HealthMetric::AVAILABILITY; // Default
}

std::string healthEventTypeToString(HealthEventType type) {
    switch (type) {
        case HealthEventType::STATUS_CHANGE: return "STATUS_CHANGE";
        case HealthEventType::THRESHOLD_BREACH: return "THRESHOLD_BREACH";
        case HealthEventType::CACHE_FAILURE: return "CACHE_FAILURE";
        case HealthEventType::CACHE_RECOVERY: return "CACHE_RECOVERY";
        case HealthEventType::PERFORMANCE_DEGRADATION: return "PERFORMANCE_DEGRADATION";
        case HealthEventType::MAINTENANCE_REQUIRED: return "MAINTENANCE_REQUIRED";
        case HealthEventType::AUTOMATIC_REPAIR: return "AUTOMATIC_REPAIR";
        default: return "UNKNOWN";
    }
}

HealthEventType stringToHealthEventType(const std::string& type_str) {
    if (type_str == "STATUS_CHANGE") return HealthEventType::STATUS_CHANGE;
    if (type_str == "THRESHOLD_BREACH") return HealthEventType::THRESHOLD_BREACH;
    if (type_str == "CACHE_FAILURE") return HealthEventType::CACHE_FAILURE;
    if (type_str == "CACHE_RECOVERY") return HealthEventType::CACHE_RECOVERY;
    if (type_str == "PERFORMANCE_DEGRADATION") return HealthEventType::PERFORMANCE_DEGRADATION;
    if (type_str == "MAINTENANCE_REQUIRED") return HealthEventType::MAINTENANCE_REQUIRED;
    if (type_str == "AUTOMATIC_REPAIR") return HealthEventType::AUTOMATIC_REPAIR;
    return HealthEventType::STATUS_CHANGE; // Default
}

bool isHealthStatusGood(HealthStatus status) {
    return status == HealthStatus::EXCELLENT || status == HealthStatus::GOOD;
}

// Global instance
CacheHealthMonitor& getGlobalCacheHealthMonitor() {
    static CacheHealthMonitor instance;
    return instance;
}

} // namespace sep::cache
