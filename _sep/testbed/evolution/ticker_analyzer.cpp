#include "ticker_analyzer.hpp"
#include <sstream>

namespace sep::testbed {
std::string TickerAnalyzer::analyze(const std::vector<Pattern>& patterns,
                                    const std::vector<Lineage>& lineage) const {
    std::ostringstream oss;
    oss << "patterns:" << patterns.size() << ";lineage:" << lineage.size();
    for (const auto& l : lineage) {
        oss << "|" << l.id;
    }
    return oss.str();
}
}
