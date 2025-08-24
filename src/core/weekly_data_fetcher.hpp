// SEP Weekly Data Fetcher
// Comprehensive data fetching system for training data preparation

#ifndef WEEKLY_DATA_FETCHER_HPP
#define WEEKLY_DATA_FETCHER_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <future>
#include <atomic>

namespace sep {
namespace train {

struct DataFetchConfig {
    std::string oanda_api_key;
    std::string oanda_account_id;
    std::string oanda_environment;  // "practice" or "live"
    std::vector<std::string> instruments;
    std::vector<std::string> granularities; // "M1", "M5", "M15", "H1", "H4", "D"
    int history_days;               // Number of days to fetch
    bool compress_data;             // Compress stored data
    int parallel_fetchers;          // Number of concurrent fetch threads
    std::string cache_dir;          // Base directory for cached candles/volume
};

struct DataFetchResult {
    std::string instrument;
    std::string granularity;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    size_t candles_fetched;
    bool success;
    std::string error_message;
    std::string cache_path;
    double fetch_duration_seconds;
};

class WeeklyDataFetcher {
public:
    WeeklyDataFetcher();
    ~WeeklyDataFetcher();

    // Configuration
    bool configure(const DataFetchConfig& config);

    // Main fetch operations
    std::vector<DataFetchResult> fetchAllInstruments();
    DataFetchResult fetchInstrument(const std::string& instrument);
    std::vector<DataFetchResult> fetchSelected(const std::vector<std::string>& instruments);

    // Status and monitoring
    bool isFetchInProgress() const;
    double getFetchProgress() const;
    std::string getCurrentOperation() const;
    std::vector<std::string> getSupportedInstruments() const;
    
private:
    DataFetchConfig config_;
    std::atomic<bool> fetch_in_progress_;
    std::atomic<double> fetch_progress_;
    mutable std::string current_operation_;
    mutable std::mutex status_mutex_;
    
    // OANDA API interface
    bool initializeOandaAPI();

    // Cache operations
    std::string getCachePath(const std::string& instrument,
                            const std::string& granularity) const;

    // Time utilities
    std::chrono::system_clock::time_point getStartTime() const;
    std::chrono::system_clock::time_point getEndTime() const;

    // Validation utilities
    bool isValidInstrument(const std::string& instrument) const;
};

// Utility functions for external use
std::vector<std::string> getStandardForexPairs();
std::vector<std::string> getStandardGranularities();

} // namespace train
} // namespace sep

#endif // WEEKLY_DATA_FETCHER_HPP
