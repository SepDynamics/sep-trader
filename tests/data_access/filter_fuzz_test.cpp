#include <gtest/gtest.h>
#include "app/filter.h"
#include <random>
#include <string>

using namespace sep::services;

TEST(FilterParsingFuzz, RandomTokens) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> lenDist(1, 30);
    const std::string chars = "abcdefghijklmnopqrstuvwxyz<>!=*&[]0123456789";
    std::uniform_int_distribution<int> charDist(0, chars.size() - 1);
    for (int i = 0; i < 100; ++i) {
        int len = lenDist(rng);
        std::string s;
        for (int j = 0; j < len; ++j) {
            s.push_back(chars[charDist(rng)]);
        }
        try {
            auto f = parse_filter(s);
            Row r{"EURUSD", sep::common::parseTimestamp("2024-01-01T00:00:00.000000Z"), 0.5, "EUR"};
            (void)match(r, f);
        } catch (const std::exception&) {
            // expected for malformed inputs
        }
    }
    SUCCEED();
}
