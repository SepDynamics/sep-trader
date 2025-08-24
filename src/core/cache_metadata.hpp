#pragma once
#include <string>
#include <chrono>

namespace sep::cache {

// Source type describing origin of cached data
enum class DataSource {
    API,
    SIMULATED,
    UNKNOWN
};

inline std::string dataSourceToString(DataSource src) {
    switch (src) {
        case DataSource::API: return "api";
        case DataSource::SIMULATED: return "simulated";
        default: return "unknown";
    }
}

inline DataSource stringToDataSource(const std::string& str) {
    if (str == "api" || str == "API") return DataSource::API;
    if (str == "simulated" || str == "SIMULATED") return DataSource::SIMULATED;
    return DataSource::UNKNOWN;
}

// Provider of market data. Currently only real providers are tracked.
enum class DataProvider {
    OANDA,
    UNKNOWN
};

inline std::string dataProviderToString(DataProvider provider) {
    switch (provider) {
        case DataProvider::OANDA: return "oanda";
        default: return "unknown";
    }
}

inline DataProvider stringToDataProvider(const std::string& str) {
    if (str == "oanda" || str == "OANDA") return DataProvider::OANDA;
    return DataProvider::UNKNOWN;
}

// Metadata stored for each cache record
struct EntryMetadata {
    std::chrono::system_clock::time_point timestamp;
    DataProvider provider{DataProvider::UNKNOWN};
};

} // namespace sep::cache

