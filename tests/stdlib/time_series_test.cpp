#include "gtest/gtest.h"
#include "util/time_series.h"
#include "util/compiler.h"
#include <vector>

using dsl::compiler::Context;
using dsl::compiler::Value;

TEST(TimeSeries, MovingAverage) {
    Context ctx;
    dsl::stdlib::register_time_series(ctx);
    auto func = ctx.get_function("moving_average");
    ASSERT_TRUE(func);
    std::vector<Value> args = {Value(1.0), Value(2.0), Value(3.0)};
    Value result = func(args);
    ASSERT_TRUE(std::holds_alternative<double>(result));
    EXPECT_DOUBLE_EQ(std::get<double>(result), 2.0);
}
