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
    bool loadConfigFromFile(const std::string& config_path);
    bool saveConfigToFile(const std::string& config_path) const;
    
    // Main fetch operations
    std::vector<DataFetchResult> fetchAllInstruments();
    DataFetchResult fetchInstrument(const std::string& instrument);
    std::vector<DataFetchResult> fetchSelected(const std::vector<std::string>& instruments);
    
    // Granularity-specific fetching
    DataFetchResult fetchInstrumentGranularity(const std::string& instrument, 
                                               const std::string& granularity);
    
    // Parallel fetching
    std::vector<DataFetchResult> fetchInstrumentsParallel(const std::vector<std::string>& instruments);
    
    // Cache management
    bool validateCachedData(const std::string& instrument) const;
    bool clearCachedData(const std::string& instrument = "");
    std::map<std::string, std::chrono::system_clock::time_point> getCacheStatus() const;
    
    // Status and monitoring
    bool isFetchInProgress() const;
    double getFetchProgress() const;
    std::string getCurrentOperation() const;
    std::vector<std::string> getSupportedInstruments() const;
    
    // Data validation
    bool validateFetchedData(const DataFetchResult& result) const;
    std::map<std::string, size_t> getDataStatistics() const;
    
private:
    DataFetchConfig config_;
    std::atomic<bool> fetch_in_progress_;
    std::atomic<double> fetch_progress_;
    mutable std::string current_operation_;
    mutable std::mutex status_mutex_;
    
    // OANDA API interface
    bool initializeOandaAPI();
    std::string makeOandaRequest(const std::string& endpoint, 
                                const std::map<std::string, std::string>& params) const;
    
    // Data processing
    bool processCandleData(const std::string& json_response,
                          const std::string& instrument,
                          const std::string& granularity,
                          const std::string& cache_path) const;
    
    // Cache operations
    std::string getCachePath(const std::string& instrument, 
                            const std::string& granularity) const;
    bool ensureCacheDirectory() const;
    bool compressDataFile(const std::string& file_path) const;
    
    // Time utilities
    std::string formatTimeForAPI(const std::chrono::system_clock::time_point& time) const;
    std::chrono::system_clock::time_point getStartTime() const;
    std::chrono::system_clock::time_point getEndTime() const;
    
    // Error handling
    void logError(const std::string& operation, const std::string& error) const;
    void logSuccess(const std::string& operation, const DataFetchResult& result) const;
    
    // Validation utilities
    bool isValidInstrument(const std::string& instrument) const;
    bool isValidGranularity(const std::string& granularity) const;
    size_t validateCandleCount(const std::string& cache_path) const;
};

// Utility functions for external use
std::vector<std::string> getStandardForexPairs();
std::vector<std::string> getStandardGranularities();
bool isMarketOpen();
std::chrono::system_clock::time_point getLastMarketClose();

} // namespace train
} // namespace sep

#endif // WEEKLY_DATA_FETCHER_HPP
