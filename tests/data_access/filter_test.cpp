#include <gtest/gtest.h>
#include "app/filter.h"
#include <random>

using namespace sep::services;

static Row make_row(const std::string& pair,
                    const std::string& ts,
                    double score,
                    const std::string& id) {
    Row r;
    r.pair = pair;
    r.ts = sep::common::parseTimestamp(ts);
    r.score = score;
    r.id = id;
    return r;
}

TEST(FilterParsing, EqualityRangeWildcard) {
    auto f = parse_filter("pair==EURUSD && score=[0.5,1.0] && id~=EUR*");
    auto r = make_row("EURUSD", "2024-01-01T00:00:00.000000Z", 0.7, "EUR123");
    EXPECT_TRUE(match(r, f));
    auto r2 = make_row("GBPUSD", "2024-01-01T00:00:00.000000Z", 0.7, "EUR123");
    EXPECT_FALSE(match(r2, f));
}

TEST(FilterParsing, Composition) {
    auto f = parse_filter("pair==EURUSD && ts>=2024-01-01 && score>=0.7 && id~=EUR*");
    auto r = make_row("EURUSD", "2024-06-01T00:00:00.000000Z", 0.8, "EUR_XYZ");
    EXPECT_TRUE(match(r, f));
    auto r2 = make_row("EURUSD", "2023-12-31T00:00:00.000000Z", 0.8, "EUR_XYZ");
    EXPECT_FALSE(match(r2, f));
}

TEST(FilterParsing, TypeCoercionFailure) {
    EXPECT_THROW(parse_filter("score==abc"), std::runtime_error);
}

TEST(FilterParsing, PerformanceOverRows) {
    auto f = parse_filter("pair==EURUSD && score>=0.5 && ts>=2024-01-01 && id~=EUR*");
    std::vector<Row> rows;
    for (int i = 0; i < 1000; ++i) {
        rows.push_back(make_row("EURUSD", "2024-01-02T00:00:00.000000Z", 0.6 + (i % 10) / 20.0, "EUR" + std::to_string(i)));
    }
    int count = 0;
    for (const auto& r : rows) {
        if (match(r, f)) ++count;
    }
    EXPECT_GT(count, 0);
}
