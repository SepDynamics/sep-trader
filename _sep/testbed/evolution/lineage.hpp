#pragma once
#include <string>
#include <vector>

namespace sep::testbed {
struct Lineage {
    std::string id;
    int generation{0};
    std::vector<std::string> parents;
    std::string mutation;
};
}
