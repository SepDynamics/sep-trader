#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include "../../_sep/testbed/perf/cuda_real_data_harness.hpp"

TEST(Performance, CUDARealData) {
    std::string path = std::string(TEST_DATA_DIR) + "/eurusd_sample.csv";
    auto data = loadRealCandleData(path);
    ASSERT_FALSE(data.empty());
    auto expected = cpuDoubleMid(data);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = gpuDoubleMid(data);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        ASSERT_DOUBLE_EQ(result[i], expected[i]);
    }
    std::cout << "CUDA real data kernel duration: " << ms << " ms\n";
}
