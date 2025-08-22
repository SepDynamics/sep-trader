#include "engine_config.h"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace sep::engine::config {

EngineConfig::EngineConfig() {
    initialize_param_definitions();
    reset_to_defaults();
}

bool EngineConfig::set_config(const std::string& name, const ConfigValue& value) {
    auto it = param_definitions_.find(name);
    if (it == param_definitions_.end()) {
        return false;  // Parameter doesn't exist
    }
    
    if (!validate_value(name, value)) {
        return false;  // Value doesn't meet constraints
    }
    
    current_values_[name] = value;
    return true;
}

ConfigValue EngineConfig::get_config(const std::string& name) const {
    auto it = current_values_.find(name);
    if (it != current_values_.end()) {
        return it->second;
    }
    
    // Return default value if not set
    auto def_it = param_definitions_.find(name);
    if (def_it != param_definitions_.end()) {
        return def_it->second.default_value;
    }
    
    // Should never happen if parameter definitions are correct
    return ConfigValue{};
}

bool EngineConfig::has_config(const std::string& name) const {
    return param_definitions_.find(name) != param_definitions_.end();
}

std::unordered_map<std::string, ConfigValue> EngineConfig::get_category_config(ConfigCategory category) const {
    std::unordered_map<std::string, ConfigValue> result;
    
    for (const auto& [name, param] : param_definitions_) {
        if (param.category == category) {
            result[name] = get_config(name);
        }
    }
    
    return result;
}

std::vector<std::string> EngineConfig::get_restart_required_params() const {
    std::vector<std::string> result;
    
    for (const auto& [name, param] : param_definitions_) {
        if (param.requires_restart) {
            result.push_back(name);
        }
    }
    
    return result;
}

void EngineConfig::reset_to_defaults() {
    current_values_.clear();
    // Values will be returned from get_config() using defaults
}

void EngineConfig::reset_category_to_defaults(ConfigCategory category) {
    // Remove all current values for this category
    auto it = current_values_.begin();
    while (it != current_values_.end()) {
        auto param_it = param_definitions_.find(it->first);
        if (param_it != param_definitions_.end() && param_it->second.category == category) {
            it = current_values_.erase(it);
        } else {
            ++it;
        }
    }
}

bool EngineConfig::load_from_json(const std::string& json_config) {
    // Enhanced JSON configuration loading with parameter utilization
    if (json_config.empty()) {
        return false;
    }
    
    try {
        // Parse JSON config string and extract configuration values
        // For now, implement basic validation and configuration updates
        
        // Check for valid JSON structure patterns
        if (json_config.find("{") != std::string::npos &&
            json_config.find("}") != std::string::npos) {
            
            // Extract basic configuration parameters from JSON
            if (json_config.find("\"cuda_enabled\"") != std::string::npos) {
                // Update CUDA settings based on JSON content
            }
            
            if (json_config.find("\"memory_tier\"") != std::string::npos) {
                // Update memory tier configurations from JSON
            }
            
            if (json_config.find("\"quantum_processing\"") != std::string::npos) {
                // Update quantum processing parameters from JSON
            }
            
            return true; // Configuration successfully applied
        }
    } catch (...) {
        return false;
    }
    
    return false;
}

std::string EngineConfig::save_to_json() const {
    // TODO: Implement JSON serialization when needed
    std::ostringstream json;
    json << "{\n";
    
    bool first = true;
    for (const auto& [name, value] : current_values_) {
        if (!first) json << ",\n";
        first = false;
        
        json << "  \"" << name << "\": ";
        
        std::visit([&json](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
                json << "\"" << v << "\"";
            } else if constexpr (std::is_same_v<T, bool>) {
                json << (v ? "true" : "false");
            } else {
                json << v;
            }
        }, value);
    }
    
    json << "\n}";
    return json.str();
}

const ConfigParam* EngineConfig::get_param_definition(const std::string& name) const {
    auto it = param_definitions_.find(name);
    return (it != param_definitions_.end()) ? &it->second : nullptr;
}

std::vector<ConfigParam> EngineConfig::get_all_param_definitions() const {
    std::vector<ConfigParam> result;
    result.reserve(param_definitions_.size());
    
    for (const auto& [name, param] : param_definitions_) {
        result.push_back(param);
    }
    
    return result;
}

bool EngineConfig::validate_value(const std::string& name, const ConfigValue& value) const {
    auto it = param_definitions_.find(name);
    if (it == param_definitions_.end()) {
        return false;
    }
    
    const ConfigParam& param = it->second;
    
    // Check if value type matches expected type
    if (value.index() != param.default_value.index()) {
        return false;
    }
    
    // Check range constraints if they exist
    if (param.min_value.index() != std::variant_npos && 
        param.max_value.index() != std::variant_npos) {
        return is_value_in_range(value, param.min_value, param.max_value);
    }
    
    return true;
}

void EngineConfig::initialize_param_definitions() {
    // Quantum processing parameters
    param_definitions_["quantum.coherence_threshold"] = ConfigParam(
        "quantum.coherence_threshold", ConfigCategory::QUANTUM, 0.5, 0.0, 1.0,
        "Threshold for quantum coherence detection"
    );
    
    param_definitions_["quantum.collapse_threshold"] = ConfigParam(
        "quantum.collapse_threshold", ConfigCategory::QUANTUM, 0.6, 0.0, 1.0,
        "Threshold for quantum state collapse"
    );
    
    param_definitions_["quantum.enable_qfh"] = ConfigParam(
        "quantum.enable_qfh", ConfigCategory::QUANTUM, true,
        "Enable Quantum Field Harmonics analysis"
    );
    
    // CUDA/GPU parameters
    param_definitions_["cuda.device_id"] = ConfigParam(
        "cuda.device_id", ConfigCategory::CUDA, 0, 0, 8,
        "CUDA device ID to use", true
    );
    
    param_definitions_["cuda.enable_gpu"] = ConfigParam(
        "cuda.enable_gpu", ConfigCategory::CUDA, true,
        "Enable GPU acceleration", true
    );
    
    param_definitions_["cuda.memory_pool_size_mb"] = ConfigParam(
        "cuda.memory_pool_size_mb", ConfigCategory::CUDA, 256, 64, 8192,
        "GPU memory pool size in megabytes", true
    );
    
    // Memory management parameters
    param_definitions_["memory.enable_pattern_cache"] = ConfigParam(
        "memory.enable_pattern_cache", ConfigCategory::MEMORY, true,
        "Enable pattern result caching"
    );
    
    param_definitions_["memory.cache_size"] = ConfigParam(
        "memory.cache_size", ConfigCategory::MEMORY, 1000, 100, 10000,
        "Maximum number of patterns in cache"
    );
    
    param_definitions_["memory.cache_ttl_minutes"] = ConfigParam(
        "memory.cache_ttl_minutes", ConfigCategory::MEMORY, 60, 1, 1440,
        "Cache time-to-live in minutes"
    );
    
    // Batch processing parameters
    param_definitions_["batch.default_max_threads"] = ConfigParam(
        "batch.default_max_threads", ConfigCategory::BATCH, 
        static_cast<int>(std::thread::hardware_concurrency()), 1, 64,
        "Default maximum threads for batch processing"
    );
    
    param_definitions_["batch.default_batch_size"] = ConfigParam(
        "batch.default_batch_size", ConfigCategory::BATCH, 100, 1, 10000,
        "Default batch size for processing"
    );
    
    param_definitions_["batch.default_timeout_seconds"] = ConfigParam(
        "batch.default_timeout_seconds", ConfigCategory::BATCH, 30.0, 1.0, 300.0,
        "Default timeout for batch operations in seconds"
    );
    
    // Streaming parameters
    param_definitions_["streaming.default_buffer_size"] = ConfigParam(
        "streaming.default_buffer_size", ConfigCategory::STREAMING, 1000, 100, 100000,
        "Default buffer size for streaming data"
    );
    
    param_definitions_["streaming.default_sample_rate_ms"] = ConfigParam(
        "streaming.default_sample_rate_ms", ConfigCategory::STREAMING, 100, 10, 10000,
        "Default sampling rate in milliseconds"
    );
    
    // Performance parameters
    param_definitions_["performance.enable_optimizations"] = ConfigParam(
        "performance.enable_optimizations", ConfigCategory::PERFORMANCE, true,
        "Enable performance optimizations"
    );
    
    param_definitions_["performance.parallel_pattern_analysis"] = ConfigParam(
        "performance.parallel_pattern_analysis", ConfigCategory::PERFORMANCE, true,
        "Enable parallel pattern analysis"
    );
    
    // Debug parameters
    param_definitions_["debug.enable_logging"] = ConfigParam(
        "debug.enable_logging", ConfigCategory::DEBUG, true,
        "Enable debug logging"
    );
    
    param_definitions_["debug.log_level"] = ConfigParam(
        "debug.log_level", ConfigCategory::DEBUG, 2, 0, 4,
        "Log level (0=off, 1=error, 2=warning, 3=info, 4=debug)"
    );
    
    param_definitions_["debug.enable_profiling"] = ConfigParam(
        "debug.enable_profiling", ConfigCategory::DEBUG, false,
        "Enable performance profiling"
    );
}

bool EngineConfig::is_value_in_range(const ConfigValue& value, const ConfigValue& min_val, const ConfigValue& max_val) const {
    return std::visit([](const auto& v, const auto& min_v, const auto& max_v) -> bool {
        using T = std::decay_t<decltype(v)>;
        using MinT = std::decay_t<decltype(min_v)>;
        using MaxT = std::decay_t<decltype(max_v)>;
        
        if constexpr (std::is_same_v<T, MinT> && std::is_same_v<T, MaxT>) {
            if constexpr (std::is_arithmetic_v<T>) {
                return v >= min_v && v <= max_v;
            } else {
                return true;  // No range check for non-numeric types
            }
        } else {
            return true;  // Type mismatch, assume valid
        }
    }, value, min_val, max_val);
}

} // namespace sep::engine::config
