#include "config_watcher.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

namespace sep::config {

ConfigWatcher::ConfigWatcher() 
    : last_event_time_{std::chrono::system_clock::now()}
{
    watching_start_time_ = std::chrono::system_clock::now();
}

ConfigWatcher::~ConfigWatcher() {
    stopWatching();
    cleanupNativeWatching();
}

bool ConfigWatcher::startWatching() {
    if (is_watching_.load()) {
        return true; // Already watching
    }
    
    stop_requested_.store(false);
    
    // Initialize native watching if available
    if (!polling_mode_.load()) {
        if (!initializeNativeWatching()) {
            // Fall back to polling mode
            polling_mode_.store(true);
            addError("Native file watching not available, using polling mode");
        }
    }
    
    // Start watch thread
    watch_thread_ = std::make_unique<std::thread>(&ConfigWatcher::watchLoop, this);
    is_watching_.store(true);
    watching_start_time_ = std::chrono::system_clock::now();
    
    return true;
}

void ConfigWatcher::stopWatching() {
    if (!is_watching_.load()) {
        return;
    }
    
    stop_requested_.store(true);
    
    if (watch_thread_ && watch_thread_->joinable()) {
        watch_thread_->join();
        watch_thread_.reset();
    }
    
    is_watching_.store(false);
    cleanupNativeWatching();
}

bool ConfigWatcher::isWatching() const {
    return is_watching_.load();
}

bool ConfigWatcher::addWatchPath(const std::string& path, bool recursive) {
    if (!isValidWatchPath(path)) {
        addError("Invalid watch path: " + path);
        return false;
    }
    
    std::lock_guard<std::mutex> lock(paths_mutex_);
    
    std::string normalized_path = normalizeWatchPath(path);
    
    // Check if path is already watched
    if (std::find(watch_paths_.begin(), watch_paths_.end(), normalized_path) != watch_paths_.end()) {
        return true; // Already watching
    }
    
    watch_paths_.push_back(normalized_path);
    path_recursive_map_[normalized_path] = recursive;
    
    // Add native watch if watching is active
    if (is_watching_.load() && !polling_mode_.load()) {
        addNativeWatch(normalized_path, recursive);
    }
    
    return true;
}

bool ConfigWatcher::removeWatchPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(paths_mutex_);
    
    std::string normalized_path = normalizeWatchPath(path);
    
    auto it = std::find(watch_paths_.begin(), watch_paths_.end(), normalized_path);
    if (it == watch_paths_.end()) {
        return false;
    }
    
    watch_paths_.erase(it);
    path_recursive_map_.erase(normalized_path);
    
    // Remove native watch if watching is active
    if (is_watching_.load() && !polling_mode_.load()) {
        removeNativeWatch(normalized_path);
    }
    
    return true;
}

std::vector<std::string> ConfigWatcher::getWatchPaths() const {
    std::lock_guard<std::mutex> lock(paths_mutex_);
    return watch_paths_;
}

bool ConfigWatcher::isPathWatched(const std::string& path) const {
    std::lock_guard<std::mutex> lock(paths_mutex_);
    
    std::string normalized_path = normalizeWatchPath(path);
    return std::find(watch_paths_.begin(), watch_paths_.end(), normalized_path) != watch_paths_.end();
}

size_t ConfigWatcher::addFileChangeCallback(FileChangeCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    change_callbacks_.push_back(callback);
    return change_callbacks_.size() - 1;
}

void ConfigWatcher::removeFileChangeCallback(size_t callback_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    if (callback_id < change_callbacks_.size()) {
        change_callbacks_.erase(change_callbacks_.begin() + callback_id);
    }
}

size_t ConfigWatcher::getCallbackCount() const {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    return change_callbacks_.size();
}

void ConfigWatcher::addFileExtensionFilter(const std::string& extension) {
    std::lock_guard<std::mutex> lock(filters_mutex_);
    
    std::string ext = extension;
    if (!ext.empty() && ext[0] != '.') {
        ext = "." + ext;
    }
    
    if (std::find(extension_filters_.begin(), extension_filters_.end(), ext) == extension_filters_.end()) {
        extension_filters_.push_back(ext);
    }
}

void ConfigWatcher::removeFileExtensionFilter(const std::string& extension) {
    std::lock_guard<std::mutex> lock(filters_mutex_);
    
    std::string ext = extension;
    if (!ext.empty() && ext[0] != '.') {
        ext = "." + ext;
    }
    
    auto it = std::find(extension_filters_.begin(), extension_filters_.end(), ext);
    if (it != extension_filters_.end()) {
        extension_filters_.erase(it);
    }
}

void ConfigWatcher::clearFileExtensionFilters() {
    std::lock_guard<std::mutex> lock(filters_mutex_);
    extension_filters_.clear();
}

std::vector<std::string> ConfigWatcher::getFileExtensionFilters() const {
    std::lock_guard<std::mutex> lock(filters_mutex_);
    return extension_filters_;
}

void ConfigWatcher::enablePollingMode(bool enable) {
    bool was_polling = polling_mode_.exchange(enable);
    
    if (was_polling != enable && is_watching_.load()) {
        // Restart watching with new mode
        stopWatching();
        startWatching();
    }
}

bool ConfigWatcher::isPollingModeEnabled() const {
    return polling_mode_.load();
}

void ConfigWatcher::setPollingInterval(std::chrono::milliseconds interval) {
    polling_interval_ = interval;
}

std::chrono::milliseconds ConfigWatcher::getPollingInterval() const {
    return polling_interval_;
}

void ConfigWatcher::enableEventHistory(bool enable, size_t max_events) {
    event_history_enabled_.store(enable);
    max_history_size_ = max_events;
    
    if (!enable) {
        clearEventHistory();
    }
}

bool ConfigWatcher::isEventHistoryEnabled() const {
    return event_history_enabled_.load();
}

std::vector<FileChangeEvent> ConfigWatcher::getEventHistory() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    return event_history_;
}

std::vector<FileChangeEvent> ConfigWatcher::getEventHistoryForPath(const std::string& path) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    std::vector<FileChangeEvent> filtered_events;
    
    for (const auto& event : event_history_) {
        if (event.path == path) {
            filtered_events.push_back(event);
        }
    }
    
    return filtered_events;
}

void ConfigWatcher::clearEventHistory() {
    std::lock_guard<std::mutex> lock(history_mutex_);
    event_history_.clear();
}

void ConfigWatcher::enableDebouncing(bool enable) {
    debouncing_enabled_.store(enable);
}

bool ConfigWatcher::isDebouncingEnabled() const {
    return debouncing_enabled_.load();
}

void ConfigWatcher::setDebounceInterval(std::chrono::milliseconds interval) {
    debounce_interval_ = interval;
}

std::chrono::milliseconds ConfigWatcher::getDebounceInterval() const {
    return debounce_interval_;
}

size_t ConfigWatcher::getTotalEventsProcessed() const {
    return total_events_processed_.load();
}

std::chrono::system_clock::time_point ConfigWatcher::getLastEventTime() const {
    return last_event_time_.load();
}

std::chrono::duration<double> ConfigWatcher::getWatchingDuration() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now - watching_start_time_);
}

std::vector<std::string> ConfigWatcher::getLastErrors() const {
    std::lock_guard<std::mutex> lock(errors_mutex_);
    return recent_errors_;
}

void ConfigWatcher::clearErrors() {
    std::lock_guard<std::mutex> lock(errors_mutex_);
    recent_errors_.clear();
}

void ConfigWatcher::watchLoop() {
    while (!stop_requested_.load()) {
        try {
            if (polling_mode_.load()) {
                processPollingEvents();
            } else {
                processNativeEvents();
            }
            
            // Sleep for a short interval
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
        } catch (const std::exception& e) {
            addError("Error in watch loop: " + std::string(e.what()));
        }
    }
}

void ConfigWatcher::processNativeEvents() {
    // Platform-specific native event processing
    // For now, fall back to polling
    processPollingEvents();
}

void ConfigWatcher::processPollingEvents() {
    std::vector<std::string> paths_copy;
    {
        std::lock_guard<std::mutex> lock(paths_mutex_);
        paths_copy = watch_paths_;
    }
    
    for (const auto& path : paths_copy) {
        try {
            if (isDirectory(path)) {
                bool recursive = false;
                {
                    std::lock_guard<std::mutex> lock(paths_mutex_);
                    auto it = path_recursive_map_.find(path);
                    if (it != path_recursive_map_.end()) {
                        recursive = it->second;
                    }
                }
                scanDirectory(path, recursive);
            } else {
                // Single file
                auto current_time = getFileModificationTime(path);
                
                std::lock_guard<std::mutex> lock(timestamps_mutex_);
                auto& last_time = file_timestamps_[path];
                
                if (current_time > last_time) {
                    last_time = current_time;
                    
                    FileChangeEvent event(FileChangeType::MODIFIED, path);
                    processFileChangeEvent(event);
                }
            }
        } catch (const std::exception& e) {
            addError("Error processing path " + path + ": " + e.what());
        }
    }
    
    // Sleep for polling interval
    std::this_thread::sleep_for(polling_interval_);
}

void ConfigWatcher::processFileChangeEvent(const FileChangeEvent& event) {
    if (!shouldProcessEvent(event)) {
        return;
    }
    
    // Update statistics
    updateStatistics(event);
    
    // Add to history if enabled
    if (event_history_enabled_.load()) {
        addToHistory(event);
    }
    
    // Notify callbacks
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    for (const auto& callback : change_callbacks_) {
        try {
            callback(event);
        } catch (const std::exception& e) {
            addError("Error in file change callback: " + std::string(e.what()));
        }
    }
}

bool ConfigWatcher::shouldProcessEvent(const FileChangeEvent& event) const {
    // Check if file is filtered
    if (isFileFiltered(event.path)) {
        return false;
    }
    
    // Check if event is debounced
    if (isDebouncedEvent(event.path)) {
        return false;
    }
    
    return true;
}

bool ConfigWatcher::isFileFiltered(const std::string& path) const {
    std::lock_guard<std::mutex> lock(filters_mutex_);
    
    // Check extension filters
    if (!extension_filters_.empty()) {
        if (!matchesExtensionFilter(path)) {
            return true; // File is filtered out
        }
    }
    
    // Check filename filters
    if (!filename_filters_.empty()) {
        if (!matchesFilenameFilter(path)) {
            return true; // File is filtered out
        }
    }
    
    return false; // File passes all filters
}

bool ConfigWatcher::isDebouncedEvent(const std::string& path) const {
    if (!debouncing_enabled_.load()) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(debounce_mutex_);
    
    auto now = std::chrono::system_clock::now();
    auto it = last_event_times_.find(path);
    
    if (it != last_event_times_.end()) {
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second);
        if (time_diff < debounce_interval_) {
            return true; // Event is debounced
        }
    }
    
    // Update last event time (const_cast is safe here as we're modifying timing data)
    const_cast<std::unordered_map<std::string, std::chrono::system_clock::time_point>&>(last_event_times_)[path] = now;
    
    return false;
}

bool ConfigWatcher::initializeNativeWatching() {
    // Platform-specific initialization
    // For this implementation, we'll use polling mode
    return false;
}

void ConfigWatcher::cleanupNativeWatching() {
    // Platform-specific cleanup
    // Nothing to do for polling mode
}

bool ConfigWatcher::addNativeWatch(const std::string& /*path*/, bool /*recursive*/) {
    // Platform-specific watch addition
    // Not implemented for polling mode
    return false;
}

bool ConfigWatcher::removeNativeWatch(const std::string& /*path*/) {
    // Platform-specific watch removal
    // Not implemented for polling mode
    return false;
}

void ConfigWatcher::scanDirectory(const std::string& directory, bool recursive) {
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            std::string path = entry.path().string();
            
            if (entry.is_regular_file()) {
                auto current_time = getFileModificationTime(path);
                
                std::lock_guard<std::mutex> lock(timestamps_mutex_);
                auto& last_time = file_timestamps_[path];
                
                if (current_time > last_time) {
                    last_time = current_time;
                    
                    FileChangeEvent event(FileChangeType::MODIFIED, path);
                    processFileChangeEvent(event);
                }
            } else if (entry.is_directory() && recursive) {
                scanDirectory(path, recursive);
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        addError("Filesystem error scanning directory " + directory + ": " + e.what());
    } catch (const std::exception& e) {
        addError("Error scanning directory " + directory + ": " + e.what());
    }
}

std::chrono::system_clock::time_point ConfigWatcher::getFileModificationTime(const std::string& path) const {
    try {
        auto ftime = std::filesystem::last_write_time(path);
        
        // Convert to system_clock time_point
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now()
        );
        
        return sctp;
    } catch (const std::exception&) {
        return std::chrono::system_clock::time_point{};
    }
}

bool ConfigWatcher::isDirectory(const std::string& path) const {
    try {
        return std::filesystem::is_directory(path);
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> ConfigWatcher::listDirectory(const std::string& directory) const {
    std::vector<std::string> entries;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            entries.push_back(entry.path().string());
        }
    } catch (const std::exception& e) {
        const_cast<ConfigWatcher*>(this)->addError("Error listing directory " + directory + ": " + e.what());
    }
    
    return entries;
}

void ConfigWatcher::addToHistory(const FileChangeEvent& event) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    event_history_.push_back(event);
    
    // Maintain maximum history size
    if (event_history_.size() > max_history_size_) {
        event_history_.erase(event_history_.begin());
    }
}

void ConfigWatcher::updateStatistics(const FileChangeEvent& event) {
    total_events_processed_.fetch_add(1);
    last_event_time_.store(event.timestamp);
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    path_event_counts_[event.path]++;
}

void ConfigWatcher::addError(const std::string& error) {
    std::lock_guard<std::mutex> lock(errors_mutex_);
    
    recent_errors_.push_back(error);
    
    // Maintain maximum error history
    if (recent_errors_.size() > MAX_STORED_ERRORS) {
        recent_errors_.erase(recent_errors_.begin());
    }
    
    // Also log to stderr for debugging
    std::cerr << "ConfigWatcher Error: " << error << std::endl;
}

bool ConfigWatcher::matchesExtensionFilter(const std::string& path) const {
    std::string extension = getFileExtension(path);
    
    return std::find(extension_filters_.begin(), extension_filters_.end(), extension) != extension_filters_.end();
}

bool ConfigWatcher::matchesFilenameFilter(const std::string& path) const {
    std::string filename = getFileName(path);
    
    for (const auto& pattern : filename_filters_) {
        // Simple pattern matching (could be enhanced with regex)
        if (filename.find(pattern) != std::string::npos) {
            return true;
        }
    }
    
    return filename_filters_.empty(); // If no filters, match all
}

std::string ConfigWatcher::getFileExtension(const std::string& path) const {
    size_t last_dot = path.find_last_of('.');
    if (last_dot != std::string::npos && last_dot < path.length() - 1) {
        return path.substr(last_dot);
    }
    return "";
}

std::string ConfigWatcher::getFileName(const std::string& path) const {
    size_t last_slash = path.find_last_of("/\\");
    if (last_slash != std::string::npos && last_slash < path.length() - 1) {
        return path.substr(last_slash + 1);
    }
    return path;
}

// Utility functions
std::string getFileChangeTypeString(FileChangeType type) {
    switch (type) {
        case FileChangeType::CREATED: return "CREATED";
        case FileChangeType::MODIFIED: return "MODIFIED";
        case FileChangeType::DELETED: return "DELETED";
        case FileChangeType::MOVED: return "MOVED";
        case FileChangeType::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

bool isValidWatchPath(const std::string& path) {
    if (path.empty()) return false;
    
    try {
        return std::filesystem::exists(path);
    } catch (const std::exception&) {
        return false;
    }
}

std::string normalizeWatchPath(const std::string& path) {
    try {
        return std::filesystem::canonical(path).string();
    } catch (const std::exception&) {
        return path;
    }
}

// Global watcher instance
ConfigWatcher& getGlobalConfigWatcher() {
    static ConfigWatcher instance;
    return instance;
}

} // namespace sep::config
