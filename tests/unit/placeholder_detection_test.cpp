#include <gtest/gtest.h>

#include <cstdlib>

#include "../../_sep/testbed/placeholder_detection.h"

TEST(PlaceholderDetectionTest, AllowsRealValue)
{
    setenv("SEP_STRICT_PLACEHOLDER_CHECK", "1", 1);
    EXPECT_NO_THROW(sep::testbed::ensure_not_placeholder("REAL"));
    unsetenv("SEP_STRICT_PLACEHOLDER_CHECK");
}

TEST(PlaceholderDetectionTest, RejectsPlaceholder)
{
    setenv("SEP_STRICT_PLACEHOLDER_CHECK", "1", 1);
    EXPECT_THROW(sep::testbed::ensure_not_placeholder("PLACEHOLDER"), std::runtime_error);
    unsetenv("SEP_STRICT_PLACEHOLDER_CHECK");
}

TEST(PlaceholderDetectionTest, DisabledCheckIgnoresPlaceholder)
{
    unsetenv("SEP_STRICT_PLACEHOLDER_CHECK");
    EXPECT_NO_THROW(sep::testbed::ensure_not_placeholder("PLACEHOLDER"));
}
