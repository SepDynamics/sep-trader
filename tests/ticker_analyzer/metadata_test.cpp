#include <gtest/gtest.h>
#include "../../_sep/testbed/evolution/ticker_analyzer.hpp"
#include "../../_sep/testbed/evolution/lineage.hpp"
#include "../../_sep/testbed/evolution/pattern.hpp"

TEST(TickerAnalyzer, SurfacesEvolutionMetadata) {
    sep::testbed::TickerAnalyzer analyzer;
    std::vector<sep::testbed::Pattern> patterns{{"p1", {1}, 0.5}};
    std::vector<sep::testbed::Lineage> lineage{{"id1", 1, {"p0"}, "flip"}};
    auto report = analyzer.analyze(patterns, lineage);
    EXPECT_NE(report.find("id1"), std::string::npos);
}
