#pragma once
#include <string>

namespace sep::cache {

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

} // namespace sep::cache
