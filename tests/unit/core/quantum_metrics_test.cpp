#include <gtest/gtest.h>

extern "C" double sep_get_total_patterns_processed();

TEST(QuantumMetricsTest, ReturnsNonNegative) {
    double value = sep_get_total_patterns_processed();
    EXPECT_GE(value, 0.0);
}
