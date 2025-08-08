#include "pair_manager.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

#include "../nlohmann_json_protected.h"

namespace sep::core {

PairManager::PairManager() : state_file_path_("/sep/config/pair_states.json") {
    initializeDefaultPairs();
    loadState();
}

PairManager::~PairManager() {
    saveState();
}

bool PairManager::addPair(const std::string& symbol) {
    if (!validatePairSymbol(symbol)) {
        return false;
    }
    
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    if (pairs_.find(symbol) != pairs_.end()) {
        return false; // Pair already exists
    }
    
    auto pair_info = std::make_unique<PairInfo>();
    pair_info->symbol = symbol;
    pair_info->status = PairStatus::UNTRAINED;
    pair_info->last_updated = std::chrono::system_clock::now();
    pair_info->enabled = false;
    
    pairs_[symbol] = std::move(pair_info);
    saveState();
    
    return true;
}

bool PairManager::removePair(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    // Don't allow removal of currently trading pairs
    if (it->second->trading_active.load()) {
        return false;
    }
    
    pairs_.erase(it);
    saveState();
    
    return true;
}

bool PairManager::enablePair(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    // Only enable if pair is ready or trading
    if (it->second->status != PairStatus::READY && it->second->status != PairStatus::TRADING) {
        return false;
    }
    
    bool was_enabled = it->second->enabled;
    it->second->enabled = true;
    it->second->last_updated = std::chrono::system_clock::now();
    
    if (!was_enabled) {
        saveState();
    }
    
    return true;
}

bool PairManager::disablePair(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    bool was_enabled = it->second->enabled;
    it->second->enabled = false;
    it->second->last_updated = std::chrono::system_clock::now();
    
    // Stop trading if currently active
    if (it->second->trading_active.load()) {
        it->second->trading_active.store(false);
        setPairStatus(symbol, PairStatus::DISABLED);
    }
    
    if (was_enabled) {
        saveState();
    }
    
    return true;
}

bool PairManager::setPairStatus(const std::string& symbol, PairStatus status) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    PairStatus old_status = it->second->status;
    it->second->status = status;
    it->second->last_updated = std::chrono::system_clock::now();
    
    // Clear error message if status is no longer ERROR
    if (status != PairStatus::ERROR) {
        it->second->error_message.clear();
    }
    
    lock.unlock(); // Unlock before callback to avoid deadlock
    
    if (old_status != status) {
        notifyStateChange(symbol, old_status, status);
        saveState();
    }
    
    return true;
}

PairStatus PairManager::getPairStatus(const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return PairStatus::ERROR;
    }
    
    return it->second->status;
}

const PairInfo& PairManager::getPairInfo(const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        // Create a static error instance to return by reference
        static PairInfo error_info;
        error_info.status = PairStatus::ERROR;
        error_info.error_message = "Pair not found";
        return error_info;
    }
    
    return *(it->second);
}

std::vector<std::string> PairManager::getAllPairs() const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    std::vector<std::string> result;
    result.reserve(pairs_.size());
    
    for (const auto& pair : pairs_) {
        result.push_back(pair.first);
    }
    
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::string> PairManager::getPairsByStatus(PairStatus status) const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    std::vector<std::string> result;
    
    for (const auto& pair : pairs_) {
        if (pair.second->status == status) {
            result.push_back(pair.first);
        }
    }
    
    std::sort(result.begin(), result.end());
    return result;
}

bool PairManager::startTrading(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    // Can only start trading if pair is ready and enabled
    if (it->second->status != PairStatus::READY || !it->second->enabled) {
        return false;
    }
    
    bool expected = false;
    if (!it->second->trading_active.compare_exchange_strong(expected, true)) {
        return false; // Already trading
    }
    
    it->second->status = PairStatus::TRADING;
    it->second->last_updated = std::chrono::system_clock::now();
    
    return true;
}

bool PairManager::stopTrading(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    bool expected = true;
    if (!it->second->trading_active.compare_exchange_strong(expected, false)) {
        return false; // Not currently trading
    }
    
    it->second->status = PairStatus::READY;
    it->second->last_updated = std::chrono::system_clock::now();
    
    return true;
}

bool PairManager::isTrading(const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    return it->second->trading_active.load();
}

bool PairManager::updateModel(const std::string& symbol, const std::string& model_path, double accuracy) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    it->second->model_path = model_path;
    it->second->accuracy = accuracy;
    it->second->last_trained = std::chrono::system_clock::now();
    it->second->last_updated = std::chrono::system_clock::now();
    
    // If previously untrained or training, mark as ready
    if (it->second->status == PairStatus::UNTRAINED || it->second->status == PairStatus::TRAINING) {
        PairStatus old_status = it->second->status;
        it->second->status = PairStatus::READY;
        
        lock.unlock();
        notifyStateChange(symbol, old_status, PairStatus::READY);
    }
    
    saveState();
    return true;
}

bool PairManager::validateModel(const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    // Check if model path exists and accuracy is reasonable
    if (it->second->model_path.empty() || it->second->accuracy <= 0.0) {
        return false;
    }
    
    // Could add actual file existence check here
    return true;
}

bool PairManager::setError(const std::string& symbol, const std::string& error_message) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    PairStatus old_status = it->second->status;
    it->second->status = PairStatus::ERROR;
    it->second->error_message = error_message;
    it->second->last_updated = std::chrono::system_clock::now();
    
    // Stop trading if active
    it->second->trading_active.store(false);
    
    lock.unlock();
    notifyStateChange(symbol, old_status, PairStatus::ERROR);
    saveState();
    
    return true;
}

bool PairManager::clearError(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(pairs_mutex_);
    
    auto it = pairs_.find(symbol);
    if (it == pairs_.end()) {
        return false;
    }
    
    if (it->second->status != PairStatus::ERROR) {
        return false;
    }
    
    PairStatus old_status = it->second->status;
    it->second->error_message.clear();
    it->second->last_updated = std::chrono::system_clock::now();
    
    // Determine appropriate new status
    PairStatus new_status = validateModel(symbol) ? PairStatus::READY : PairStatus::UNTRAINED;
    it->second->status = new_status;
    
    lock.unlock();
    notifyStateChange(symbol, old_status, new_status);
    saveState();
    
    return true;
}

void PairManager::addStateChangeCallback(StateChangeCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    state_callbacks_.push_back(callback);
}

void PairManager::removeStateChangeCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (callback_id < state_callbacks_.size()) {
        state_callbacks_.erase(state_callbacks_.begin() + callback_id);
    }
}

size_t PairManager::getTotalPairs() const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    return pairs_.size();
}

size_t PairManager::getActivePairs() const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    size_t active = 0;
    for (const auto& pair : pairs_) {
        if (pair.second->trading_active.load()) {
            active++;
        }
    }
    return active;
}

size_t PairManager::getTrainingPairs() const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    size_t training = 0;
    for (const auto& pair : pairs_) {
        if (pair.second->status == PairStatus::TRAINING) {
            training++;
        }
    }
    return training;
}

double PairManager::getAverageAccuracy() const {
    std::shared_lock<std::shared_mutex> lock(pairs_mutex_);
    
    double total_accuracy = 0.0;
    size_t count = 0;
    
    for (const auto& pair : pairs_) {
        if (pair.second->accuracy > 0.0) {
            total_accuracy += pair.second->accuracy;
            count++;
        }
    }
    
    return count > 0 ? total_accuracy / count : 0.0;
}

void PairManager::notifyStateChange(const std::string& symbol, PairStatus old_status, PairStatus new_status) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& callback : state_callbacks_) {
        try {
            callback(symbol, old_status, new_status);
        } catch (const std::exception& e) {
            // Log error but continue with other callbacks
            std::cerr << "Error in state change callback: " << e.what() << std::endl;
        }
    }
}

bool PairManager::validatePairSymbol(const std::string& symbol) const {
    return isValidPairSymbol(symbol);
}

void PairManager::initializeDefaultPairs() {
    // Initialize with common currency pairs
    std::vector<std::string> default_pairs = {
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF",
        "EUR_GBP", "EUR_JPY", "EUR_AUD", "GBP_JPY", "AUD_JPY", "NZD_USD"
    };
    
    for (const auto& symbol : default_pairs) {
        auto pair_info = std::make_unique<PairInfo>();
        pair_info->symbol = symbol;
        pair_info->status = PairStatus::UNTRAINED;
        pair_info->last_updated = std::chrono::system_clock::now();
        pair_info->enabled = false;
        
        pairs_[symbol] = std::move(pair_info);
    }
}

bool PairManager::saveState() {
    try {
        std::string json_data = serializeState();
        std::ofstream file(state_file_path_);
        if (!file.is_open()) {
            return false;
        }
        file << json_data;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving pair state: " << e.what() << std::endl;
        return false;
    }
}

bool PairManager::loadState() {
    try {
        std::ifstream file(state_file_path_);
        if (!file.is_open()) {
            return false; // File doesn't exist yet, use defaults
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_data = buffer.str();
        
        return deserializeState(json_data);
    } catch (const std::exception& e) {
        std::cerr << "Error loading pair state: " << e.what() << std::endl;
        return false;
    }
}

std::string PairManager::serializeState() const {
    std::stringstream ss;
    ss << "{\n  \"pairs\": {\n";
    
    bool first = true;
    for (const auto& pair : pairs_) {
        if (!first) ss << ",\n";
        first = false;
        
        const auto& info = pair.second;
        auto time_t = std::chrono::system_clock::to_time_t(info->last_updated);
        
        ss << "    \"" << pair.first << "\": {\n";
        ss << "      \"status\": \"" << statusToString(info->status) << "\",\n";
        ss << "      \"enabled\": " << (info->enabled ? "true" : "false") << ",\n";
        ss << "      \"accuracy\": " << info->accuracy << ",\n";
        ss << "      \"model_path\": \"" << info->model_path << "\",\n";
        ss << "      \"error_message\": \"" << info->error_message << "\",\n";
        ss << "      \"last_updated\": " << time_t << "\n";
        ss << "    }";
    }
    
    ss << "\n  }\n}";
    return ss.str();
}

bool PairManager::deserializeState(const std::string& json_data) {
    try {
        nlohmann::json json = nlohmann::json::parse(json_data);
        
        std::unordered_map<std::string, std::unique_ptr<PairInfo>> loaded_pairs;
        
        // Check if "pairs" key exists and iterate over it
        if (json.contains("pairs") && json["pairs"].is_object()) {
            for (auto& [symbol, pair_data] : json["pairs"].items()) {
            auto pair_info = std::make_unique<PairInfo>();
            pair_info->symbol = symbol;
            pair_info->status = stringToStatus(pair_data["status"].get<std::string>());
            pair_info->enabled = pair_data["enabled"].get<bool>();
            pair_info->accuracy = pair_data["accuracy"].get<double>();
            pair_info->model_path = pair_data["model_path"].get<std::string>();
            pair_info->error_message = pair_data["error_message"].get<std::string>();
            pair_info->last_updated = std::chrono::system_clock::now();
            
                loaded_pairs[symbol] = std::move(pair_info);
            }
        }
        
        // Only update if we successfully parsed at least some pairs
        if (!loaded_pairs.empty()) {
            pairs_ = std::move(loaded_pairs);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON state: " << e.what() << std::endl;
        return false;
    }
}

// Utility functions
std::string statusToString(sep::core::PairStatus status) {
    switch (status) {
        case sep::core::PairStatus::UNTRAINED: return "UNTRAINED";
        case sep::core::PairStatus::TRAINING: return "TRAINING";
        case sep::core::PairStatus::READY: return "READY";
        case sep::core::PairStatus::TRADING: return "TRADING";
        case sep::core::PairStatus::DISABLED: return "DISABLED";
        case sep::core::PairStatus::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

PairStatus stringToStatus(const std::string& status_str) {
    if (status_str == "UNTRAINED") return PairStatus::UNTRAINED;
    if (status_str == "TRAINING") return PairStatus::TRAINING;
    if (status_str == "READY") return PairStatus::READY;
    if (status_str == "TRADING") return PairStatus::TRADING;
    if (status_str == "DISABLED") return PairStatus::DISABLED;
    if (status_str == "ERROR") return PairStatus::ERROR;
    return PairStatus::ERROR;
}

bool isValidPairSymbol(const std::string& symbol) {
    // Validate currency pair format: CCC_CCC (3 letters, underscore, 3 letters)
    std::regex pattern(R"([A-Z]{3}_[A-Z]{3})");
    return std::regex_match(symbol, pattern);
}

} // namespace sep::core
