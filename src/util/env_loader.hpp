#pragma once
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>

namespace sep {
namespace util {

/**
 * Simple .env file loader utility
 * Loads key=value pairs from environment files like config/OANDA.env
 */
class EnvLoader {
public:
    /**
     * Load environment variables from a file
     * @param filepath Path to the .env file
     * @return Map of key-value pairs
     */
    static std::unordered_map<std::string, std::string> loadFromFile(const std::string& filepath);
    
    /**
     * Get environment variable with fallback to system environment
     * @param key Variable name
     * @param envVars Map from loadFromFile
     * @return Variable value or empty string if not found
     */
    static std::string getEnvVar(const std::string& key, const std::unordered_map<std::string, std::string>& envVars);
};

} // namespace util
} // namespace sep