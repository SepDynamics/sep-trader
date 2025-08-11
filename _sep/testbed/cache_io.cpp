#include "cache_io.hpp"

#include <filesystem>
#include <fstream>
#include <chrono>

namespace sep::testbed {

namespace fs = std::filesystem;

static std::string current_iso_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

bool write_cache_with_provenance(const std::string& path, const CacheFile& cache) {
    fs::create_directories(fs::path(path).parent_path());
    nlohmann::json root;
    root["data"] = cache.data;
    root["provenance"] = {
        {"source", cache.provenance.source},
        {"retrieved_at", cache.provenance.retrieved_at.empty() ? current_iso_time() : cache.provenance.retrieved_at},
        {"revision", cache.provenance.revision}
    };
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }
    out << root.dump(2);
    return true;
}

CacheFile read_cache_with_provenance(const std::string& path) {
    CacheFile result;
    std::ifstream in(path);
    if (!in.is_open()) {
        return result;
    }
    nlohmann::json root;
    in >> root;
    if (root.contains("data") && root["data"].is_array()) {
        for (const auto& item : root["data"]) {
            result.data.push_back(item.is_string() ? item.get<std::string>() : item.dump());
        }
    }
    if (root.contains("provenance")) {
        auto prov = root["provenance"];
        result.provenance.source = prov.value("source", "");
        result.provenance.retrieved_at = prov.value("retrieved_at", "");
        result.provenance.revision = prov.value("revision", "");
    }
    return result;
}

} // namespace sep::testbed
