#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include "../../_sep/testbed/cuda_marketdata_harness.hpp"

TEST(Performance, CUDAValidation) {
    std::vector<sep::connectors::MarketData> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].mid = static_cast<double>(i);
    }
    auto expected = cpuDoubleMid(data);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = gpuDoubleMid(data);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        ASSERT_DOUBLE_EQ(result[i], expected[i]);
    }
    std::cout << "CUDA kernel duration: " << ms << " ms\n";
}
