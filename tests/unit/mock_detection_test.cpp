#include <gtest/gtest.h>

#include <cstdlib>

#include "../../_sep/testbed/mock_detection.h"

TEST(MockDetectionTest, AllowsRealData)
{
    setenv("SEP_STRICT_MOCK_CHECK", "1", 1);
    sep::core::CandleData candle;
    EXPECT_NO_THROW(sep::testbed::ensure_not_mock(candle));
    unsetenv("SEP_STRICT_MOCK_CHECK");
}

TEST(MockDetectionTest, RejectsMockData)
{
    setenv("SEP_STRICT_MOCK_CHECK", "1", 1);
    sep::core::CandleData candle;
    candle.is_mock = true;
    EXPECT_THROW(sep::testbed::ensure_not_mock(candle), std::runtime_error);
    unsetenv("SEP_STRICT_MOCK_CHECK");
}

TEST(MockDetectionTest, DisabledCheckIgnoresMock)
{
    unsetenv("SEP_STRICT_MOCK_CHECK");
    sep::core::CandleData candle;
    candle.is_mock = true;
    EXPECT_NO_THROW(sep::testbed::ensure_not_mock(candle));
}
