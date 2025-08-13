#pragma once

#include <string>
#include <vector>
#include "nlohmann_json_safe.h"

namespace sep::testbed {

struct CacheProvenance {
    std::string source;
    std::string retrieved_at; // ISO-8601 string
    std::string revision;
};

struct CacheFile {
    std::vector<std::string> data;
    CacheProvenance provenance;
};

bool write_cache_with_provenance(const std::string& path, const CacheFile& cache);
CacheFile read_cache_with_provenance(const std::string& path);

} // namespace sep::testbed
