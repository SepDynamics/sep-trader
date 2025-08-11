#include <nlohmann/json.hpp>
#include "dynamic_config_manager.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <iomanip>
#include <chrono>
#include "engine/internal/standard_includes.h"

namespace sep::config {

DynamicConfigManager::DynamicConfigManager() 
    : last_modified_{std::chrono::system_clock::now()},
      last_reload_{std::chrono::system_clock::now()}
{
}

DynamicConfigManager::~DynamicConfigManager() {
    stopWatching();
}

bool DynamicConfigManager::loadConfiguration(const std::string& config_path) {
    std::unique_lock<std::shared_mutex> lock(config_mutex_);
    
    config_file_path_ = config_path;
    
    std::unordered_map<std::string, std::string> new_config;
    if (!parseJsonFile(config_path, new_config)) {
        return false;
    }
    
    // Store old config for change detection
    auto old_config = config_data_;
    config_data_ = std::move(new_config);
    
    last_reload_.store(std::chrono::system_clock::now());
    reload_count_.fetch_add(1);
    
    lock.unlock();
    
    // Notify of reload event
    ConfigChangeEvent reload_event(ConfigChangeType::RELOADED, "");
    notifyConfigChange(reload_event);
    
    // Start watching if enabled
    if (hot_reload_enabled_.load()) {
        addWatchPath(config_path);
        startWatching();
    }
    
    return true;
}

bool DynamicConfigManager::reloadConfiguration() {
    if (config_file_path_.empty()) {
        return false;
    }
    
    return loadConfiguration(config_file_path_);
}

bool DynamicConfigManager::saveConfiguration(const std::string& config_path) {
    std::shared_lock<std::shared_mutex> lock(config_mutex_);
    
    std::string path = config_path.empty() ? config_file_path_ : config_path;
    if (path.empty()) {
        return false;
    }
    
    return writeJsonFile(path, config_data_);
}

std::string DynamicConfigManager::getString(const std::string& key, const std::string& default_value) const {
    std::shared_lock<std::shared_mutex> lock(config_mutex_);
    
    auto it = config_data_.find(key);
    return (it != config_data_.end()) ? it->second : default_value;
}

int DynamicConfigManager::getInt(const std::string& key, int default_value) const {
    std::string value = getString(key);
    if (value.empty() || !isNumeric(value)) {
        return default_value;
    }
    
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        return default_value;
    }
}

double DynamicConfigManager::getDouble(const std::string& key, double default_value) const {
    std::string value = getString(key);
    if (value.empty() || !isNumeric(value)) {
        return default_value;
    }
    
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        return default_value;
    }
}

bool DynamicConfigManager::getBool(const std::string& key, bool default_value) const {
    std::string value = getString(key);
    if (value.empty() || !isBooleanString(value)) {
        return default_value;
    }
    
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return value == "true" || value == "1" || value == "yes" || value == "on";
}

bool DynamicConfigManager::setString(const std::string& key, const std::string& value) {
    if (!isValidConfigKey(key)) {
        return false;
    }
    
    if (!runValidation(key, value)) {
        return false;
    }
    
    std::unique_lock<std::shared_mutex> lock(config_mutex_);
    
    std::string old_value;
    ConfigChangeType change_type = ConfigChangeType::ADDED;
    
    auto it = config_data_.find(key);
    if (it != config_data_.end()) {
        old_value = it->second;
        change_type = ConfigChangeType::MODIFIED;
    }
    
    config_data_[key] = value;
    last_modified_.store(std::chrono::system_clock::now());
    change_count_.fetch_add(1);
    
    lock.unlock();
    
    // Notify change
    ConfigChangeEvent change_event(change_type, key, old_value, value);
    notifyConfigChange(change_event);
    
    return true;
}

bool DynamicConfigManager::setInt(const std::string& key, int value) {
    return setString(key, toString(value));
}

bool DynamicConfigManager::setDouble(const std::string& key, double value) {
    return setString(key, toString(value));
}

bool DynamicConfigManager::setBool(const std::string& key, bool value) {
    return setString(key, toString(value));
}

bool DynamicConfigManager::hasKey(const std::string& key) const {
    std::shared_lock<std::shared_mutex> lock(config_mutex_);
    return config_data_.find(key) != config_data_.end();
}

bool DynamicConfigManager::removeKey(const std::string& key) {
    std::unique_lock<std::shared_mutex> lock(config_mutex_);
    
    auto it = config_data_.find(key);
    if (it == config_data_.end()) {
        return false;
    }
    
    std::string old_value = it->second;
    config_data_.erase(it);
    last_modified_.store(std::chrono::system_clock::now());
    change_count_.fetch_add(1);
    
    lock.unlock();
    
    // Notify removal
    ConfigChangeEvent change_event(ConfigChangeType::REMOVED, key, old_value, "");
    notifyConfigChange(change_event);
    
    return true;
}

std::vector<std::string> DynamicConfigManager::getAllKeys() const {
    std::shared_lock<std::shared_mutex> lock(config_mutex_);
    
    std::vector<std::string> keys;
    keys.reserve(config_data_.size());
    
    for (const auto& pair : config_data_) {
        keys.push_back(pair.first);
    }
    
    std::sort(keys.begin(), keys.end());
    return keys;
}

std::vector<std::string> DynamicConfigManager::getKeysWithPrefix(const std::string& prefix) const {
    std::shared_lock<std::shared_mutex> lock(config_mutex_);
    
    std::vector<std::string> keys;
    
    for (const auto& pair : config_data_) {
        if (pair.first.substr(0, prefix.length()) == prefix) {
            keys.push_back(pair.first);
        }
    }
    
    std::sort(keys.begin(), keys.end());
    return keys;
}

void DynamicConfigManager::enableHotReload(bool enable) {
    hot_reload_enabled_.store(enable);
    
    if (enable && !config_file_path_.empty()) {
        startWatching();
    } else if (!enable) {
        stopWatching();
    }
}

bool DynamicConfigManager::isHotReloadEnabled() const {
    return hot_reload_enabled_.load();
}

void DynamicConfigManager::addWatchPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(watch_mutex_);
    
    if (std::find(watch_paths_.begin(), watch_paths_.end(), path) == watch_paths_.end()) {
        watch_paths_.push_back(path);
    }
}

void DynamicConfigManager::removeWatchPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(watch_mutex_);
    
    auto it = std::find(watch_paths_.begin(), watch_paths_.end(), path);
    if (it != watch_paths_.end()) {
        watch_paths_.erase(it);
    }
}

std::vector<std::string> DynamicConfigManager::getWatchPaths() const {
    std::lock_guard<std::mutex> lock(watch_mutex_);
    return watch_paths_;
}

size_t DynamicConfigManager::addChangeCallback(ConfigChangeCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    change_callbacks_.push_back(callback);
    return change_callbacks_.size() - 1;
}

void DynamicConfigManager::removeChangeCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    if (callback_id < change_callbacks_.size()) {
        change_callbacks_.erase(change_callbacks_.begin() + callback_id);
    }
}

bool DynamicConfigManager::validateConfiguration() const {
    std::shared_lock<std::shared_mutex> lock(config_mutex_);
    
    for (const auto& pair : config_data_) {
        if (!runValidation(pair.first, pair.second)) {
            return false;
        }
    }
    
    return true;
}

void DynamicConfigManager::startWatching() {
    if (watch_thread_ && watch_thread_->joinable()) {
        return; // Already watching
    }
    
    stop_watching_.store(false);
    watch_thread_ = std::make_unique<std::thread>(&DynamicConfigManager::watchFiles, this);
}

void DynamicConfigManager::stopWatching() {
    stop_watching_.store(true);
    watch_cv_.notify_all();
    
    if (watch_thread_ && watch_thread_->joinable()) {
        watch_thread_->join();
        watch_thread_.reset();
    }
}

void DynamicConfigManager::watchFiles() {
    std::unordered_map<std::string, std::chrono::system_clock::time_point> last_modified_times;
    
    // Initialize modification times
    {
        std::lock_guard<std::mutex> lock(watch_mutex_);
        for (const auto& path : watch_paths_) {
            last_modified_times[path] = getFileModificationTime(path);
        }
    }
    
    while (!stop_watching_.load()) {
        std::unique_lock<std::mutex> lock(watch_mutex_);
        
        // Wait for interval or stop signal
        watch_cv_.wait_for(lock, reload_interval_, [this] { return stop_watching_.load(); });
        
        if (stop_watching_.load()) {
            break;
        }
        
        // Check each watched path
        std::vector<std::string> paths_copy = watch_paths_;
        lock.unlock();
        
        for (const auto& path : paths_copy) {
            auto current_time = getFileModificationTime(path);
            auto& last_time = last_modified_times[path];
            
            if (current_time > last_time) {
                last_time = current_time;
                
                // File was modified, reload configuration
                if (path == config_file_path_) {
                    reloadConfiguration();
                }
            }
        }
    }
}

std::chrono::system_clock::time_point DynamicConfigManager::getFileModificationTime(const std::string& path) const {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            return std::chrono::system_clock::time_point{};
        }
        
        // Simple approach - check if file can be opened
        // In production, would use platform-specific file stat calls
        return std::chrono::system_clock::now();
    } catch (const std::exception&) {
        return std::chrono::system_clock::time_point{};
    }
}

void DynamicConfigManager::notifyConfigChange(const ConfigChangeEvent& event) {
    if (history_enabled_.load()) {
        addToHistory(event);
    }
    
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : change_callbacks_) {
        try {
            callback(event);
        } catch (const std::exception& e) {
            // Log error but continue with other callbacks
            std::cerr << "Error in config change callback: " << e.what() << std::endl;
        }
    }
}

bool DynamicConfigManager::runValidation(const std::string& key, const std::string& value) const {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& validation_pair : validation_callbacks_) {
        const std::string& pattern = validation_pair.first;
        const auto& validator = validation_pair.second;
        
        // Simple pattern matching (key prefix)
        if (key.substr(0, pattern.length()) == pattern) {
            try {
                if (!validator(key, value)) {
                    return false;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in validation callback: " << e.what() << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

void DynamicConfigManager::addToHistory(const ConfigChangeEvent& event) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    change_history_.push_back(event);
    
    // Maintain maximum history size
    if (change_history_.size() > max_history_size_) {
        change_history_.erase(change_history_.begin());
    }
}

bool DynamicConfigManager::parseJsonFile(const std::string& file_path, std::unordered_map<std::string, std::string>& config) const {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return false;
        }
        
        nlohmann::json json_data;
        file >> json_data;
        
        // Convert JSON to string map - flatten nested objects
        std::function<void(const nlohmann::json&, const std::string&)> flatten;
        flatten = [&](const nlohmann::json& obj, const std::string& prefix) {
            for (auto& [key, value] : obj.items()) {
                std::string full_key = prefix.empty() ? key : prefix + "." + key;
                
                if (value.is_object()) {
                    flatten(value, full_key);
                } else if (value.is_string()) {
                    config[full_key] = value.get<std::string>();
                } else if (value.is_number()) {
                    config[full_key] = value.dump();
                } else if (value.is_boolean()) {
                    config[full_key] = value.get<bool>() ? "true" : "false";
                } else {
                    config[full_key] = value.dump();
                }
            }
        };
        
        flatten(json_data, "");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
        return false;
    }
}

bool DynamicConfigManager::writeJsonFile(const std::string& file_path, const std::unordered_map<std::string, std::string>& config) const {
    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            return false;
        }
        
        file << "{\n";
        bool first = true;
        for (const auto& pair : config) {
            if (!first) file << ",\n";
            first = false;
            
            file << "  \"" << pair.first << "\": ";
            
            // Simple type detection
            if (isBooleanString(pair.second)) {
                file << pair.second;
            } else if (isNumeric(pair.second)) {
                file << pair.second;
            } else {
                file << "\"" << pair.second << "\"";
            }
        }
        file << "\n}";
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing JSON file: " << e.what() << std::endl;
        return false;
    }
}

// Helper methods
bool DynamicConfigManager::isNumeric(const std::string& str) const {
    if (str.empty()) return false;
    
    bool has_decimal = false;
    size_t start = 0;
    
    if (str[0] == '-' || str[0] == '+') {
        start = 1;
    }
    
    for (size_t i = start; i < str.length(); ++i) {
        if (str[i] == '.') {
            if (has_decimal) return false;
            has_decimal = true;
        } else if (!std::isdigit(str[i])) {
            return false;
        }
    }
    
    return start < str.length();
}

bool DynamicConfigManager::isBooleanString(const std::string& str) const {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    
    return lower_str == "true" || lower_str == "false" || 
           lower_str == "1" || lower_str == "0" ||
           lower_str == "yes" || lower_str == "no" ||
           lower_str == "on" || lower_str == "off";
}

std::string DynamicConfigManager::toString(int value) const {
    return std::to_string(value);
}

std::string DynamicConfigManager::toString(double value) const {
    return std::to_string(value);
}

std::string DynamicConfigManager::toString(bool value) const {
    return value ? "true" : "false";
}

// Global instance
DynamicConfigManager& getGlobalConfigManager() {
    static DynamicConfigManager instance;
    return instance;
}

// Utility functions
std::string getConfigChangeTypeString(ConfigChangeType type) {
    switch (type) {
        case ConfigChangeType::ADDED: return "ADDED";
        case ConfigChangeType::MODIFIED: return "MODIFIED";
        case ConfigChangeType::REMOVED: return "REMOVED";
        case ConfigChangeType::RELOADED: return "RELOADED";
        default: return "UNKNOWN";
    }
}

bool isValidConfigKey(const std::string& key) {
    if (key.empty()) return false;
    
    // Allow alphanumeric, dots, underscores, and hyphens
    std::regex pattern(R"([a-zA-Z0-9._-]+)");
    return std::regex_match(key, pattern);
}

std::string sanitizeConfigKey(const std::string& key) {
    std::string sanitized;
    for (char c : key) {
        if (std::isalnum(c) || c == '.' || c == '_' || c == '-') {
            sanitized += c;
        } else {
            sanitized += '_';
        }
    }
    return sanitized;
}

} // namespace sep::config
