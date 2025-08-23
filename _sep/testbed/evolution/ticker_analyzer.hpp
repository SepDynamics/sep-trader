#pragma once
#include "pattern.hpp"
#include "lineage.hpp"
#include <string>
#include <vector>

namespace sep::testbed {
class TickerAnalyzer {
public:
    std::string analyze(const std::vector<Pattern>& patterns,
                         const std::vector<Lineage>& lineage) const;
};
}
