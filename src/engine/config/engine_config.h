#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <memory>
#include <vector>
#include <thread>

namespace sep::engine::config {

/**
 * @brief Configuration value type that can hold different data types
 */
using ConfigValue = std::variant<bool, int, double, std::string>;

/**
 * @brief Engine configuration categories
 */
enum class ConfigCategory {
    QUANTUM,        // Quantum processing settings
    CUDA,          // CUDA/GPU settings  
    MEMORY,        // Memory management settings
    BATCH,         // Batch processing settings
    STREAMING,     // Streaming data settings
    CACHE,         // Pattern cache settings
    PERFORMANCE,   // Performance tuning settings
    DEBUG          // Debug and logging settings
};

/**
 * @brief Configuration parameter definition
 */
struct ConfigParam {
    std::string name;
    ConfigCategory category;
    ConfigValue default_value;
    ConfigValue min_value;
    ConfigValue max_value;
    std::string description;
    bool requires_restart = false;  // Whether changing this requires engine restart
    
    ConfigParam(const std::string& n, ConfigCategory cat, ConfigValue def, 
                const std::string& desc, bool restart = false)
        : name(n), category(cat), default_value(def), description(desc), requires_restart(restart) {}
        
    ConfigParam(const std::string& n, ConfigCategory cat, ConfigValue def,
                ConfigValue min_val, ConfigValue max_val, const std::string& desc, bool restart = false)
        : name(n), category(cat), default_value(def), min_value(min_val), max_value(max_val), 
          description(desc), requires_restart(restart) {}
};

/**
 * @brief Engine configuration manager
 */
class EngineConfig {
public:
    EngineConfig();
    ~EngineConfig() = default;
    
    /**
     * @brief Set a configuration parameter
     * @param name Parameter name (e.g., "quantum.coherence_threshold")
     * @param value New value
     * @return true if successfully set, false if parameter doesn't exist or value is invalid
     */
    bool set_config(const std::string& name, const ConfigValue& value);
    
    /**
     * @brief Get a configuration parameter
     * @param name Parameter name
     * @return Current value, or default if not set
     */
    ConfigValue get_config(const std::string& name) const;
    
    /**
     * @brief Get configuration parameter as specific type
     */
    template<typename T>
    T get_config_as(const std::string& name) const {
        auto value = get_config(name);
        return std::get<T>(value);
    }
    
    /**
     * @brief Check if a parameter exists
     */
    bool has_config(const std::string& name) const;
    
    /**
     * @brief Get all parameters in a category
     */
    std::unordered_map<std::string, ConfigValue> get_category_config(ConfigCategory category) const;
    
    /**
     * @brief Get all parameters that require restart after change
     */
    std::vector<std::string> get_restart_required_params() const;
    
    /**
     * @brief Reset all configuration to defaults
     */
    void reset_to_defaults();
    
    /**
     * @brief Reset specific category to defaults
     */
    void reset_category_to_defaults(ConfigCategory category);
    
    /**
     * @brief Load configuration from JSON string
     */
    bool load_from_json(const std::string& json_config);
    
    /**
     * @brief Save current configuration to JSON string
     */
    std::string save_to_json() const;
    
    /**
     * @brief Get parameter definition
     */
    const ConfigParam* get_param_definition(const std::string& name) const;
    
    /**
     * @brief Get all parameter definitions
     */
    std::vector<ConfigParam> get_all_param_definitions() const;
    
    /**
     * @brief Validate a configuration value against parameter constraints
     */
    bool validate_value(const std::string& name, const ConfigValue& value) const;

private:
    std::unordered_map<std::string, ConfigParam> param_definitions_;
    std::unordered_map<std::string, ConfigValue> current_values_;
    
    /**
     * @brief Initialize all parameter definitions
     */
    void initialize_param_definitions();
    
    /**
     * @brief Validate value against min/max constraints
     */
    bool is_value_in_range(const ConfigValue& value, const ConfigValue& min_val, const ConfigValue& max_val) const;
};

} // namespace sep::engine::config
