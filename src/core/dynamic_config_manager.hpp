#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>
#include <any>
#include <chrono>
#include <mutex>

namespace sep::config {

enum class ConfigSource {
    DEFAULT,       // Default configuration values
    FILE,          // Values loaded from config file
    ENVIRONMENT,   // Values from environment variables
    COMMAND_LINE,  // Values from command line arguments
    RUNTIME,       // Values set during runtime
    REMOTE         // Values from remote configuration service
};

enum class ConfigValueType {
    STRING,
    INTEGER,
    FLOAT,
    BOOLEAN,
    OBJECT,
    ARRAY,
    NULL_VALUE
};

struct ConfigChangeEvent {
    std::string key;
    std::any old_value;
    std::any new_value;
    ConfigSource source;
    std::chrono::system_clock::time_point timestamp;
    
    ConfigChangeEvent() = default;
};

using ConfigChangeCallback = std::function<void(const ConfigChangeEvent&)>;

class DynamicConfigManager {
public:
    DynamicConfigManager();
    ~DynamicConfigManager();
    
    // Core configuration methods
    template<typename T>
    T getValue(const std::string& key, const T& default_value) const;
    
    template<typename T>
    std::optional<T> getValue(const std::string& key) const;
    
    template<typename T>
    bool setValue(const std::string& key, const T& value, ConfigSource source = ConfigSource::RUNTIME);
    
    bool hasKey(const std::string& key) const;
    bool removeKey(const std::string& key);
    void clear();
    
    // Configuration sources
    bool loadFromFile(const std::string& file_path);
    bool saveToFile(const std::string& file_path) const;
    bool loadFromEnvironment(const std::string& prefix = "SEP_");
    bool loadFromCommandLine(int argc, char** argv);
    
    // Event handling
    size_t registerChangeCallback(const std::string& key_prefix, ConfigChangeCallback callback);
    bool unregisterChangeCallback(size_t callback_id);
    
    // Utilities
    std::vector<std::string> getKeys() const;
    std::vector<std::string> getKeysByPrefix(const std::string& prefix) const;
    ConfigValueType getValueType(const std::string& key) const;
    ConfigSource getValueSource(const std::string& key) const;
    
    // Specialized for common types to avoid template instantiation issues
    std::string getStringValue(const std::string& key, const std::string& default_value = "") const;
    int getIntValue(const std::string& key, int default_value = 0) const;
    double getDoubleValue(const std::string& key, double default_value = 0.0) const;
    bool getBoolValue(const std::string& key, bool default_value = false) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Internal precedence helper
inline int configSourcePrecedence(ConfigSource source) {
    switch (source) {
        case ConfigSource::DEFAULT: return 0;
        case ConfigSource::FILE: return 1;
        case ConfigSource::ENVIRONMENT: return 2;
        case ConfigSource::COMMAND_LINE: return 3;
        case ConfigSource::RUNTIME: return 4;
        case ConfigSource::REMOTE: return 5;
    }
    return 0;
}

// Helper functions
std::string configSourceToString(ConfigSource source);
ConfigSource stringToConfigSource(const std::string& source_str);
std::string configValueTypeToString(ConfigValueType type);
ConfigValueType stringToConfigValueType(const std::string& type_str);

} // namespace sep::config