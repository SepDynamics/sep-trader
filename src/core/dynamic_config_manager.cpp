#include "core/dynamic_config_manager.hpp"
#include <map>
#include <mutex>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <unistd.h>

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

std::string DynamicConfigManager::getStringValue(const std::string& key, const std::string& default_value) const {
    return getValue<std::string>(key, default_value);
}

int DynamicConfigManager::getIntValue(const std::string& key, int default_value) const {
    return getValue<int>(key, default_value);
}

double DynamicConfigManager::getDoubleValue(const std::string& key, double default_value) const {
    return getValue<double>(key, default_value);
}

bool DynamicConfigManager::getBoolValue(const std::string& key, bool default_value) const {
    return getValue<bool>(key, default_value);
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

namespace {
// helper to parse string -> typed value
template<typename T>
T parseNumber(const std::string& s) {
    std::istringstream iss(s);
    T v{}; iss >> v; return v;
}

bool tryParseInt(const std::string& s, int& out) {
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end && *end == '\0') { out = static_cast<int>(v); return true; }
    return false;
}

bool tryParseDouble(const std::string& s, double& out) {
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end && *end == '\0') { out = v; return true; }
    return false;
}

} // namespace

bool DynamicConfigManager::loadFromFile(const std::string& file_path) {
    std::ifstream in(file_path);
    if (!in.is_open()) {
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        int iv; double dv;
        if (value == "true" || value == "false") {
            setValue<bool>(key, value == "true", ConfigSource::FILE);
        } else if (tryParseInt(value, iv)) {
            setValue<int>(key, iv, ConfigSource::FILE);
        } else if (tryParseDouble(value, dv)) {
            setValue<double>(key, dv, ConfigSource::FILE);
        } else {
            setValue<std::string>(key, value, ConfigSource::FILE);
        }
    }
    return true;
}

bool DynamicConfigManager::saveToFile(const std::string& file_path) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    std::ofstream out(file_path);
    if (!out.is_open()) {
        return false;
    }
    for (const auto& pair : impl_->config_values) {
        const std::any& v = pair.second;
        out << pair.first << '=';
        if (v.type() == typeid(std::string)) {
            out << std::any_cast<std::string>(v);
        } else if (v.type() == typeid(int)) {
            out << std::any_cast<int>(v);
        } else if (v.type() == typeid(double)) {
            out << std::any_cast<double>(v);
        } else if (v.type() == typeid(bool)) {
            out << (std::any_cast<bool>(v) ? "true" : "false");
        }
        out << '\n';
    }
    return true;
}

bool DynamicConfigManager::loadFromEnvironment(const std::string& prefix) {
    // Use system environ properly
    if (!::environ) return false;
    
    for (char **env = ::environ; *env != nullptr; ++env) {
        std::string entry(*env);
        auto pos = entry.find('=');
        if (pos == std::string::npos) continue;
        std::string key = entry.substr(0, pos);
        if (key.rfind(prefix, 0) != 0) continue;
        std::string value = entry.substr(pos + 1);
        key = key.substr(prefix.size());
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c){ return std::tolower(c); });
        int iv; double dv;
        if (value == "true" || value == "false") {
            setValue<bool>(key, value == "true", ConfigSource::ENVIRONMENT);
        } else if (tryParseInt(value, iv)) {
            setValue<int>(key, iv, ConfigSource::ENVIRONMENT);
        } else if (tryParseDouble(value, dv)) {
            setValue<double>(key, dv, ConfigSource::ENVIRONMENT);
        } else {
            setValue<std::string>(key, value, ConfigSource::ENVIRONMENT);
        }
    }
    return true;
}

bool DynamicConfigManager::loadFromCommandLine(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) continue;
        std::string key;
        std::string value;
        auto eq = arg.find('=');
        if (eq != std::string::npos) {
            key = arg.substr(2, eq - 2);
            value = arg.substr(eq + 1);
        } else {
            key = arg.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("-", 0) != 0) {
                value = argv[++i];
            } else {
                value = "true";
            }
        }
        int iv; double dv;
        if (value == "true" || value == "false") {
            setValue<bool>(key, value == "true", ConfigSource::COMMAND_LINE);
        } else if (tryParseInt(value, iv)) {
            setValue<int>(key, iv, ConfigSource::COMMAND_LINE);
        } else if (tryParseDouble(value, dv)) {
            setValue<double>(key, dv, ConfigSource::COMMAND_LINE);
        } else {
            setValue<std::string>(key, value, ConfigSource::COMMAND_LINE);
        }
    }
    return true;
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
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    auto it = impl_->config_values.find(key);
    if (it == impl_->config_values.end()) {
        return ConfigValueType::NULL_VALUE;
    }
    const std::type_info& t = it->second.type();
    if (t == typeid(std::string)) return ConfigValueType::STRING;
    if (t == typeid(int)) return ConfigValueType::INTEGER;
    if (t == typeid(double)) return ConfigValueType::FLOAT;
    if (t == typeid(bool)) return ConfigValueType::BOOLEAN;
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

// ===== Template Implementations =====
namespace sep::config {

template<typename T>
T DynamicConfigManager::getValue(const std::string& key, const T& default_value) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    auto it = impl_->config_values.find(key);
    if (it == impl_->config_values.end()) {
        return default_value;
    }
    try {
        return std::any_cast<T>(it->second);
    } catch (...) {
        return default_value;
    }
}

template<typename T>
std::optional<T> DynamicConfigManager::getValue(const std::string& key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    auto it = impl_->config_values.find(key);
    if (it == impl_->config_values.end()) {
        return std::nullopt;
    }
    try {
        return std::any_cast<T>(it->second);
    } catch (...) {
        return std::nullopt;
    }
}

template<typename T>
bool DynamicConfigManager::setValue(const std::string& key, const T& value, ConfigSource source) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    std::any old_value;
    ConfigSource old_source = ConfigSource::DEFAULT;
    auto it = impl_->config_values.find(key);
    if (it != impl_->config_values.end()) {
        old_value = it->second;
        old_source = impl_->config_sources[key];
        if (configSourcePrecedence(source) < configSourcePrecedence(old_source)) {
            return false;
        }
    }

    impl_->config_values[key] = value;
    impl_->config_sources[key] = source;

    for (const auto& cb_pair : impl_->callbacks) {
        const auto& prefix = cb_pair.second.first;
        if (key.rfind(prefix, 0) == 0) {
            ConfigChangeEvent evt;
            evt.key = key;
            evt.old_value = old_value;
            evt.new_value = value;
            evt.source = source;
            evt.timestamp = std::chrono::system_clock::now();
            cb_pair.second.second(evt);
        }
    }

    return true;
}

// ===== Explicit Template Instantiations =====
template std::string DynamicConfigManager::getValue(const std::string&, const std::string&) const;
template int DynamicConfigManager::getValue(const std::string&, const int&) const;
template double DynamicConfigManager::getValue(const std::string&, const double&) const;
template bool DynamicConfigManager::getValue(const std::string&, const bool&) const;

template std::optional<std::string> DynamicConfigManager::getValue(const std::string&) const;
template std::optional<int> DynamicConfigManager::getValue(const std::string&) const;
template std::optional<double> DynamicConfigManager::getValue(const std::string&) const;
template std::optional<bool> DynamicConfigManager::getValue(const std::string&) const;

template bool DynamicConfigManager::setValue(const std::string&, const std::string&, ConfigSource);
template bool DynamicConfigManager::setValue(const std::string&, const int&, ConfigSource);
template bool DynamicConfigManager::setValue(const std::string&, const double&, ConfigSource);
template bool DynamicConfigManager::setValue(const std::string&, const bool&, ConfigSource);

} // namespace sep::config
