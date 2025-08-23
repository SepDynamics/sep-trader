#pragma once
#include <string>
#include <vector>

namespace sep::testbed {
struct Pattern {
    std::string id;
    std::vector<uint64_t> mask;
    double fitness{0.0};
};
}
