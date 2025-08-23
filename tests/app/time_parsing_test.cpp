#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

#include "util/financial_data_types.h"

using namespace sep::common;

TEST(TimeParsing, GoldenFile)
{
    namespace fs = std::filesystem;
    fs::path path = fs::path(__FILE__).parent_path() / "fixtures" / "timestamps_golden.csv";
    std::ifstream file(path);
    ASSERT_TRUE(file.is_open()) << "missing golden file";

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto comma = line.find(',');
        ASSERT_NE(comma, std::string::npos);
        std::string ts = line.substr(0, comma);
        int64_t expected = std::stoll(line.substr(comma + 1));
        auto tp = parseTimestamp(ts);
        int64_t actual = time_point_to_nanoseconds(tp);
        EXPECT_EQ(actual, expected) << ts;
    }
}
