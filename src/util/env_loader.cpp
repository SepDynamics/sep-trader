#include "util/env_loader.hpp"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <fstream>
#include <sstream>

namespace sep {
namespace util {

std::unordered_map<std::string, std::string> EnvLoader::loadFromFile(const std::string& filepath) {
    std::unordered_map<std::string, std::string> envVars;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        // Silently return empty map if file doesn't exist
        return envVars;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Find the '=' separator
        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            continue;
        }
        
        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);
        
        // Trim whitespace from key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Remove quotes if present
        if (value.size() >= 2 && 
            ((value.front() == '"' && value.back() == '"') ||
             (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        
        envVars[key] = value;
    }
    
    return envVars;
}

std::string EnvLoader::getEnvVar(const std::string& key, const std::unordered_map<std::string, std::string>& envVars) {
    // First check the loaded environment variables from file
    auto it = envVars.find(key);
    if (it != envVars.end() && !it->second.empty()) {
        return it->second;
    }
    
    // Fallback to system environment variable
    const char* sysEnv = std::getenv(key.c_str());
    if (sysEnv) {
        return std::string(sysEnv);
    }
    
    return "";
}

} // namespace util
} // namespace sep