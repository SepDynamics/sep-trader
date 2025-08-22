#include "nlohmann_json_safe.h"
#include <iostream>
#include "src/cache/cache_metadata.hpp"

using namespace sep::cache;

bool verify_provenance(const nlohmann::json& entry) {
    if (!entry.contains("timestamp") || !entry.contains("provider")) return false;
    auto provider = stringToDataProvider(entry["provider"].get<std::string>());
    return provider == DataProvider::OANDA;
}

int main() {
    nlohmann::json good = {{"timestamp", 1}, {"provider", "oanda"}};
    nlohmann::json bad = {{"timestamp", 1}, {"provider", "stub"}};
    std::cout << verify_provenance(good) << "\n";
    std::cout << verify_provenance(bad) << "\n";
    return 0;
}
