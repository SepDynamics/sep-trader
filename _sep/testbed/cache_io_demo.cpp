#include "cache_io.hpp"
#include <iostream>

int main() {
    sep::testbed::CacheFile cache;
    cache.data = {"one", "two"};
    cache.provenance.source = "demo";
    cache.provenance.revision = "test";
    std::string path = "/_sep/testbed/demo_cache.json";
    if (!sep::testbed::write_cache_with_provenance(path, cache)) {
        std::cerr << "write failed\n";
        return 1;
    }
    auto loaded = sep::testbed::read_cache_with_provenance(path);
    std::cout << loaded.data.size() << " " << loaded.provenance.source << "\n";
    return 0;
}
