#include "dynamic_config_manager.hpp"
#include <map>
#include <mutex>

namespace sep::config {

class DynamicConfigManager::Impl {
public:
    Impl() = default;
    ~Impl() = default;
    
    std::unordered_map<std::string, std::any> config_values;
    std::unordered_map<std::string, ConfigSource> config_sources;
    std::map<size_t, std::pair<std::string, ConfigChangeCallback>> callbacks;
    size_t next_callback_id = 1;
    std::mutex mutex_;
};

DynamicConfigManager::DynamicConfigManager() : impl_(std::make_unique<Impl>()) {}
DynamicConfigManager::~DynamicConfigManager() = default;

// Template specializations need to be defined in the header or explicitly instantiated here
// For simplicity, we'll use the non-template versions for this stub implementation

std::string DynamicConfigManager::getStringValue(const std::string& key, const std::string& default_value) const {
    // This is a stub implementation
    return default_value;
}

int DynamicConfigManager::getIntValue(const std::string& key, int default_value) const {
    // This is a stub implementation
    return default_value;
}

double DynamicConfigManager::getDoubleValue(const std::string& key, double default_value) const {
    // This is a stub implementation
    return default_value;
}

bool DynamicConfigManager::getBoolValue(const std::string& key, bool default_value) const {
    // This is a stub implementation
    return default_value;
}

bool DynamicConfigManager::hasKey(const std::string& key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    return impl_->config_values.find(key) != impl_->config_values.end();
}

bool DynamicConfigManager::removeKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    auto it = impl_->config_values.find(key);
    if (it != impl_->config_values.end()) {
        impl_->config_values.erase(it);
        impl_->config_sources.erase(key);
        return true;
    }
    return false;
}

void DynamicConfigManager::clear() {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->config_values.clear();
    impl_->config_sources.clear();
}

bool DynamicConfigManager::loadFromFile(const std::string& file_path) {
    // This is a stub implementation
    return false;
}

bool DynamicConfigManager::saveToFile(const std::string& file_path) const {
    // This is a stub implementation
    return false;
}

bool DynamicConfigManager::loadFromEnvironment(const std::string& prefix) {
    // This is a stub implementation
    return false;
}

bool DynamicConfigManager::loadFromCommandLine(int argc, char** argv) {
    // This is a stub implementation
    return false;
}

size_t DynamicConfigManager::registerChangeCallback(const std::string& key_prefix, ConfigChangeCallback callback) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    size_t id = impl_->next_callback_id++;
    impl_->callbacks[id] = std::make_pair(key_prefix, callback);
    return id;
}

bool DynamicConfigManager::unregisterChangeCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    auto it = impl_->callbacks.find(callback_id);
    if (it != impl_->callbacks.end()) {
        impl_->callbacks.erase(it);
        return true;
    }
    return false;
}

std::vector<std::string> DynamicConfigManager::getKeys() const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    std::vector<std::string> keys;
    keys.reserve(impl_->config_values.size());
    for (const auto& pair : impl_->config_values) {
        keys.push_back(pair.first);
    }
    return keys;
}

std::vector<std::string> DynamicConfigManager::getKeysByPrefix(const std::string& prefix) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    std::vector<std::string> keys;
    for (const auto& pair : impl_->config_values) {
        if (pair.first.compare(0, prefix.size(), prefix) == 0) {
            keys.push_back(pair.first);
        }
    }
    return keys;
}

ConfigValueType DynamicConfigManager::getValueType(const std::string& key) const {
    // This is a stub implementation
    return ConfigValueType::NULL_VALUE;
}

ConfigSource DynamicConfigManager::getValueSource(const std::string& key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    auto it = impl_->config_sources.find(key);
    if (it != impl_->config_sources.end()) {
        return it->second;
    }
    return ConfigSource::DEFAULT;
}

std::string configSourceToString(ConfigSource source) {
    switch (source) {
        case ConfigSource::DEFAULT: return "DEFAULT";
        case ConfigSource::FILE: return "FILE";
        case ConfigSource::ENVIRONMENT: return "ENVIRONMENT";
        case ConfigSource::COMMAND_LINE: return "COMMAND_LINE";
        case ConfigSource::RUNTIME: return "RUNTIME";
        case ConfigSource::REMOTE: return "REMOTE";
        default: return "UNKNOWN";
    }
}

ConfigSource stringToConfigSource(const std::string& source_str) {
    if (source_str == "DEFAULT") return ConfigSource::DEFAULT;
    if (source_str == "FILE") return ConfigSource::FILE;
    if (source_str == "ENVIRONMENT") return ConfigSource::ENVIRONMENT;
    if (source_str == "COMMAND_LINE") return ConfigSource::COMMAND_LINE;
    if (source_str == "RUNTIME") return ConfigSource::RUNTIME;
    if (source_str == "REMOTE") return ConfigSource::REMOTE;
    return ConfigSource::DEFAULT; // Default
}

std::string configValueTypeToString(ConfigValueType type) {
    switch (type) {
        case ConfigValueType::STRING: return "STRING";
        case ConfigValueType::INTEGER: return "INTEGER";
        case ConfigValueType::FLOAT: return "FLOAT";
        case ConfigValueType::BOOLEAN: return "BOOLEAN";
        case ConfigValueType::OBJECT: return "OBJECT";
        case ConfigValueType::ARRAY: return "ARRAY";
        case ConfigValueType::NULL_VALUE: return "NULL_VALUE";
        default: return "UNKNOWN";
    }
}

ConfigValueType stringToConfigValueType(const std::string& type_str) {
    if (type_str == "STRING") return ConfigValueType::STRING;
    if (type_str == "INTEGER") return ConfigValueType::INTEGER;
    if (type_str == "FLOAT") return ConfigValueType::FLOAT;
    if (type_str == "BOOLEAN") return ConfigValueType::BOOLEAN;
    if (type_str == "OBJECT") return ConfigValueType::OBJECT;
    if (type_str == "ARRAY") return ConfigValueType::ARRAY;
    if (type_str == "NULL_VALUE") return ConfigValueType::NULL_VALUE;
    return ConfigValueType::NULL_VALUE; // Default
}

} // namespace sep::config