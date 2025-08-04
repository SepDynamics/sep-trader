#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

#ifdef __linux__
#include <sys/inotify.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <CoreServices/CoreServices.h>
#endif

namespace sep::config {

// File change event types
enum class FileChangeType {
    CREATED,    // File was created
    MODIFIED,   // File was modified
    DELETED,    // File was deleted
    MOVED,      // File was moved/renamed
    UNKNOWN     // Unknown change type
};

// File change event
struct FileChangeEvent {
    FileChangeType type;
    std::string path;
    std::string old_path;  // For move operations
    std::chrono::system_clock::time_point timestamp;
    
    FileChangeEvent(FileChangeType t, const std::string& p, const std::string& old_p = "")
        : type(t), path(p), old_path(old_p), timestamp(std::chrono::system_clock::now()) {}
};

// File change callback type
using FileChangeCallback = std::function<void(const FileChangeEvent& event)>;

class ConfigWatcher {
public:
    ConfigWatcher();
    ~ConfigWatcher();

    // Watching control
    bool startWatching();
    void stopWatching();
    bool isWatching() const;
    
    // Path management
    bool addWatchPath(const std::string& path, bool recursive = false);
    bool removeWatchPath(const std::string& path);
    std::vector<std::string> getWatchPaths() const;
    bool isPathWatched(const std::string& path) const;
    
    // Callback management
    size_t addFileChangeCallback(FileChangeCallback callback);
    void removeFileChangeCallback(size_t callback_id);
    size_t getCallbackCount() const;
    
    // Filter management
    void addFileExtensionFilter(const std::string& extension);
    void removeFileExtensionFilter(const std::string& extension);
    void clearFileExtensionFilters();
    std::vector<std::string> getFileExtensionFilters() const;
    
    void addFileNameFilter(const std::string& pattern);
    void removeFileNameFilter(const std::string& pattern);
    void clearFileNameFilters();
    std::vector<std::string> getFileNameFilters() const;
    
    // Polling mode (fallback when native watching is not available)
    void enablePollingMode(bool enable = true);
    bool isPollingModeEnabled() const;
    void setPollingInterval(std::chrono::milliseconds interval);
    std::chrono::milliseconds getPollingInterval() const;
    
    // Event history
    void enableEventHistory(bool enable = true, size_t max_events = 1000);
    bool isEventHistoryEnabled() const;
    std::vector<FileChangeEvent> getEventHistory() const;
    std::vector<FileChangeEvent> getEventHistoryForPath(const std::string& path) const;
    void clearEventHistory();
    
    // Debouncing (prevent multiple events for the same file in quick succession)
    void enableDebouncing(bool enable = true);
    bool isDebouncingEnabled() const;
    void setDebounceInterval(std::chrono::milliseconds interval);
    std::chrono::milliseconds getDebounceInterval() const;
    
    // Statistics
    size_t getTotalEventsProcessed() const;
    size_t getEventsProcessedForPath(const std::string& path) const;
    std::chrono::system_clock::time_point getLastEventTime() const;
    std::chrono::duration<double> getWatchingDuration() const;
    
    // Error handling
    std::vector<std::string> getLastErrors() const;
    void clearErrors();
    
private:
    // Core watching state
    std::atomic<bool> is_watching_{false};
    std::atomic<bool> stop_requested_{false};
    std::unique_ptr<std::thread> watch_thread_;
    
    // Path management
    std::vector<std::string> watch_paths_;
    std::unordered_map<std::string, bool> path_recursive_map_;
    mutable std::mutex paths_mutex_;
    
    // Callback management
    std::vector<FileChangeCallback> change_callbacks_;
    mutable std::mutex callbacks_mutex_;
    
    // Filter management
    std::vector<std::string> extension_filters_;
    std::vector<std::string> filename_filters_;
    mutable std::mutex filters_mutex_;
    
    // Polling mode
    std::atomic<bool> polling_mode_{false};
    std::chrono::milliseconds polling_interval_{1000}; // 1 second default
    std::unordered_map<std::string, std::chrono::system_clock::time_point> file_timestamps_;
    mutable std::mutex timestamps_mutex_;
    
    // Event history
    std::atomic<bool> event_history_enabled_{false};
    size_t max_history_size_{1000};
    std::vector<FileChangeEvent> event_history_;
    mutable std::mutex history_mutex_;
    
    // Debouncing
    std::atomic<bool> debouncing_enabled_{true};
    std::chrono::milliseconds debounce_interval_{100}; // 100ms default
    std::unordered_map<std::string, std::chrono::system_clock::time_point> last_event_times_;
    mutable std::mutex debounce_mutex_;
    
    // Statistics
    std::atomic<size_t> total_events_processed_{0};
    std::unordered_map<std::string, size_t> path_event_counts_;
    std::atomic<std::chrono::system_clock::time_point> last_event_time_;
    std::chrono::system_clock::time_point watching_start_time_;
    mutable std::mutex stats_mutex_;
    
    // Error handling
    std::vector<std::string> recent_errors_;
    mutable std::mutex errors_mutex_;
    static constexpr size_t MAX_STORED_ERRORS = 100;
    
    // Platform-specific data
#ifdef __linux__
    int inotify_fd_{-1};
    std::unordered_map<int, std::string> watch_descriptors_;
#elif defined(_WIN32)
    std::vector<HANDLE> directory_handles_;
    std::vector<std::unique_ptr<char[]>> watch_buffers_;
#elif defined(__APPLE__)
    FSEventStreamRef event_stream_{nullptr};
    CFRunLoopRef run_loop_{nullptr};
#endif

    // Core watching implementation
    void watchLoop();
    void processNativeEvents();
    void processPollingEvents();
    
    // Event processing
    void processFileChangeEvent(const FileChangeEvent& event);
    bool shouldProcessEvent(const FileChangeEvent& event) const;
    bool isFileFiltered(const std::string& path) const;
    bool isDebouncedEvent(const std::string& path) const;
    
    // Platform-specific implementations
    bool initializeNativeWatching();
    void cleanupNativeWatching();
    bool addNativeWatch(const std::string& path, bool recursive);
    bool removeNativeWatch(const std::string& path);
    
    // Polling implementation
    void scanDirectory(const std::string& directory, bool recursive);
    std::chrono::system_clock::time_point getFileModificationTime(const std::string& path) const;
    bool isDirectory(const std::string& path) const;
    std::vector<std::string> listDirectory(const std::string& directory) const;
    
    // Utility methods
    void addToHistory(const FileChangeEvent& event);
    void updateStatistics(const FileChangeEvent& event);
    void addError(const std::string& error);
    std::string getAbsolutePath(const std::string& path) const;
    bool pathExists(const std::string& path) const;
    
    // Filter matching
    bool matchesExtensionFilter(const std::string& path) const;
    bool matchesFilenameFilter(const std::string& path) const;
    std::string getFileExtension(const std::string& path) const;
    std::string getFileName(const std::string& path) const;
};

// Utility functions
std::string getFileChangeTypeString(FileChangeType type);
bool isValidWatchPath(const std::string& path);
std::string normalizeWatchPath(const std::string& path);

// Global watcher instance
ConfigWatcher& getGlobalConfigWatcher();

} // namespace sep::config
