#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

TEST(AccuracyBenchmark, CalculatesDatasetAccuracy)
{
    std::ifstream file(std::string(TEST_DATA_DIR) + "/eurusd_sample.csv");
    ASSERT_TRUE(file.is_open()) << "Unable to open dataset";

    std::string line;
    std::getline(file, line);  // skip header

    size_t total = 0;
    size_t up = 0;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string date, open, high, low, close;
        std::getline(ss, date, ',');
        std::getline(ss, open, ',');
        std::getline(ss, high, ',');
        std::getline(ss, low, ',');
        std::getline(ss, close, ',');

        double o = std::stod(open);
        double c = std::stod(close);
        if (c > o)
        {
            ++up;
        }
        ++total;
    }

    ASSERT_GT(total, 0u);
    double accuracy = static_cast<double>(up) / static_cast<double>(total);
    EXPECT_NEAR(0.6, accuracy, 1e-6);
}
