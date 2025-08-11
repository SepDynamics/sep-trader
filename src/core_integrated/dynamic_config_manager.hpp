#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include "engine/internal/standard_includes.h"
#include <atomic>
#include <thread>
#include <condition_variable>

namespace sep::config {

// Configuration change event types
enum class ConfigChangeType {
    ADDED,      // New configuration item added
    MODIFIED,   // Existing configuration item changed
    REMOVED,    // Configuration item removed
    RELOADED    // Entire configuration reloaded
};

// Configuration change event
struct ConfigChangeEvent {
    ConfigChangeType type;
    std::string key;
    std::string old_value;
    std::string new_value;
    std::chrono::system_clock::time_point timestamp;
    
    ConfigChangeEvent(ConfigChangeType t, const std::string& k, 
                     const std::string& old_val = "", const std::string& new_val = "")
        : type(t), key(k), old_value(old_val), new_value(new_val), 
          timestamp(std::chrono::system_clock::now()) {}
};

// Configuration change callback type
using ConfigChangeCallback = std::function<void(const ConfigChangeEvent& event)>;

// Configuration validation callback type
using ConfigValidationCallback = std::function<bool(const std::string& key, const std::string& value)>;

class DynamicConfigManager {
public:
    DynamicConfigManager();
    ~DynamicConfigManager();

    // Configuration management
    bool loadConfiguration(const std::string& config_path);
    bool reloadConfiguration();
    bool saveConfiguration(const std::string& config_path = "");
    
    // Value access (with type conversion)
    std::string getString(const std::string& key, const std::string& default_value = "") const;
    int getInt(const std::string& key, int default_value = 0) const;
    double getDouble(const std::string& key, double default_value = 0.0) const;
    bool getBool(const std::string& key, bool default_value = false) const;
    
    // Value modification
    bool setString(const std::string& key, const std::string& value);
    bool setInt(const std::string& key, int value);
    bool setDouble(const std::string& key, double value);
    bool setBool(const std::string& key, bool value);
    
    // Key management
    bool hasKey(const std::string& key) const;
    bool removeKey(const std::string& key);
    std::vector<std::string> getAllKeys() const;
    std::vector<std::string> getKeysWithPrefix(const std::string& prefix) const;
    
    // Hot-reload management
    void enableHotReload(bool enable = true);
    bool isHotReloadEnabled() const;
    void setReloadInterval(std::chrono::milliseconds interval);
    std::chrono::milliseconds getReloadInterval() const;
    
    // File watching
    void addWatchPath(const std::string& path);
    void removeWatchPath(const std::string& path);
    std::vector<std::string> getWatchPaths() const;
    
    // Event system
    size_t addChangeCallback(ConfigChangeCallback callback);
    void removeChangeCallback(size_t callback_id);
    size_t addValidationCallback(const std::string& key_pattern, ConfigValidationCallback callback);
    void removeValidationCallback(size_t callback_id);
    
    // Configuration validation
    bool validateConfiguration() const;
    bool validateKey(const std::string& key, const std::string& value) const;
    std::vector<std::string> getValidationErrors() const;
    
    // Configuration history
    void enableHistory(bool enable = true, size_t max_history = 100);
    bool isHistoryEnabled() const;
    std::vector<ConfigChangeEvent> getHistory() const;
    std::vector<ConfigChangeEvent> getHistoryForKey(const std::string& key) const;
    
    // Configuration sections
    std::unordered_map<std::string, std::string> getSection(const std::string& section_prefix) const;
    bool setSection(const std::string& section_prefix, const std::unordered_map<std::string, std::string>& values);
    bool removeSection(const std::string& section_prefix);
    
    // Atomic operations
    bool atomicUpdate(const std::unordered_map<std::string, std::string>& updates);
    bool atomicUpdateSection(const std::string& section_prefix, const std::unordered_map<std::string, std::string>& values);
    
    // Configuration backup and restore
    bool createBackup(const std::string& backup_name = "");
    bool restoreFromBackup(const std::string& backup_name);
    std::vector<std::string> listBackups() const;
    bool deleteBackup(const std::string& backup_name);
    
    // Configuration merging
    bool mergeConfiguration(const std::string& config_path, bool overwrite_existing = false);
    bool mergeConfigurationData(const std::unordered_map<std::string, std::string>& config_data, bool overwrite_existing = false);
    
    // Statistics and monitoring
    size_t getConfigurationSize() const;
    std::chrono::system_clock::time_point getLastModified() const;
    std::chrono::system_clock::time_point getLastReload() const;
    size_t getReloadCount() const;
    size_t getChangeCount() const;
    
    // Thread safety
    std::shared_lock<std::shared_mutex> getReadLock() const;
    std::unique_lock<std::shared_mutex> getWriteLock();
    
private:
    mutable std::shared_mutex config_mutex_;
    std::unordered_map<std::string, std::string> config_data_;
    std::string config_file_path_;
    
    // Hot reload settings
    std::atomic<bool> hot_reload_enabled_{true};
    std::chrono::milliseconds reload_interval_{1000}; // 1 second default
    std::vector<std::string> watch_paths_;
    
    // File watching thread
    std::unique_ptr<std::thread> watch_thread_;
    std::atomic<bool> stop_watching_{false};
    std::condition_variable watch_cv_;
    mutable std::mutex watch_mutex_;
    
    // Event system
    std::vector<ConfigChangeCallback> change_callbacks_;
    std::vector<std::pair<std::string, ConfigValidationCallback>> validation_callbacks_;
    mutable std::mutex callbacks_mutex_;
    
    // History system
    std::atomic<bool> history_enabled_{false};
    size_t max_history_size_{100};
    std::vector<ConfigChangeEvent> change_history_;
    mutable std::mutex history_mutex_;
    
    // Statistics
    std::atomic<std::chrono::system_clock::time_point> last_modified_;
    std::atomic<std::chrono::system_clock::time_point> last_reload_;
    std::atomic<size_t> reload_count_{0};
    std::atomic<size_t> change_count_{0};
    
    // Internal methods
    void startWatching();
    void stopWatching();
    void watchFiles();
    bool checkFileModified(const std::string& path) const;
    std::chrono::system_clock::time_point getFileModificationTime(const std::string& path) const;
    
    void notifyConfigChange(const ConfigChangeEvent& event);
    bool runValidation(const std::string& key, const std::string& value) const;
    void addToHistory(const ConfigChangeEvent& event);
    
    // JSON parsing helpers
    bool parseJsonFile(const std::string& file_path, std::unordered_map<std::string, std::string>& config) const;
    bool writeJsonFile(const std::string& file_path, const std::unordered_map<std::string, std::string>& config) const;
    std::string flattenJsonKey(const std::string& prefix, const std::string& key) const;
    
    // Backup management
    std::string generateBackupName() const;
    std::string getBackupPath(const std::string& backup_name) const;
    
    // Type conversion helpers
    bool isNumeric(const std::string& str) const;
    bool isBooleanString(const std::string& str) const;
    std::string toString(int value) const;
    std::string toString(double value) const;
    std::string toString(bool value) const;
};

// Global configuration manager instance
DynamicConfigManager& getGlobalConfigManager();

// Utility functions
std::string getConfigChangeTypeString(ConfigChangeType type);
bool isValidConfigKey(const std::string& key);
std::string sanitizeConfigKey(const std::string& key);

} // namespace sep::config
