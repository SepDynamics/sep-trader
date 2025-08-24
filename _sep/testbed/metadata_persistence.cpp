#include "nlohmann_json_safe.h"
#include <fstream>
#include <iostream>
#include "../src/core/cache_metadata.hpp"

int main() {
    using namespace sep::cache;
    nlohmann::json root;
    root["data"] = nlohmann::json::array();
    root["data"].push_back({{"timestamp",1}, {"provider", dataProviderToString(DataProvider::OANDA)}});
    root["data"].push_back({{"timestamp",2}, {"provider", dataProviderToString(DataProvider::OANDA)}});
    std::ofstream("_sep/testbed/test_cache.json") << root.dump();

    nlohmann::json loaded;
    std::ifstream("_sep/testbed/test_cache.json") >> loaded;
    bool ok = true;
    for (const auto& entry : loaded["data"]) {
        if (!entry.contains("provider")) ok = false;
    }
    std::cout << (ok ? "metadata_ok" : "metadata_missing") << std::endl;
    return ok ? 0 : 1;
}
