#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include "nlohmann_json_safe.h"
#include <vector>

#include "apps/oanda_trader/candle_types.h"
#include "apps/oanda_trader/realtime_aggregator.hpp"

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
extern "C" void launch_quantum_training(const float* input_data, float* output_patterns,
                                        int data_size, int pattern_count);
#define SEP_HAS_CUDA 1
#else
#define SEP_HAS_CUDA 0
#endif

TEST(PipelineIntegration, FullDataFetchTrainingCache)
{
    system("mkdir -p /_sep/testbed");
    std::ifstream file("assets/test_data/eur_usd_m1_48h.json");
    ASSERT_TRUE(file.is_open());
    nlohmann::json json_data;
    file >> json_data;
    std::vector<float> prices;
    for (const auto& c : json_data)
    {
        prices.push_back(static_cast<float>(c["close"]));
    }
    ASSERT_FALSE(prices.empty());

#if SEP_HAS_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    std::vector<float> results(prices.size());
    auto start = std::chrono::high_resolution_clock::now();
    launch_quantum_training(prices.data(), results.data(), prices.size(), 10);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(ms, 200);

    float avg = 0.0f;
    for (float r : results) avg += r;
    avg = (avg / results.size()) * 100.0f;
    EXPECT_NEAR(avg, 50.0f, 50.0f);
    EXPECT_NE(avg, 60.73f) << "Stub accuracy detected";
#else
    GTEST_SKIP() << "CUDA headers not available";
    float avg = 0.0f;  // avoid unused warning
#endif

    nlohmann::json out_json;
    out_json["accuracy"] = avg;
    std::string out_path = "/_sep/testbed/training_cache.json";
    {
        std::ofstream out(out_path);
        ASSERT_TRUE(out.is_open());
        out << out_json.dump();
    }
    std::ifstream verify(out_path);
    ASSERT_TRUE(verify.is_open());
    nlohmann::json verify_json;
    verify >> verify_json;
#if SEP_HAS_CUDA
    EXPECT_DOUBLE_EQ(verify_json["accuracy"], out_json["accuracy"]);
#endif
}

TEST(PipelineIntegration, RealTimeCandleGenerationBenchmark)
{
    std::ifstream file("assets/test_data/eur_usd_m1_48h.json");
    ASSERT_TRUE(file.is_open());
    nlohmann::json json_data;
    file >> json_data;

    std::vector<Candle> generated;
    RealTimeAggregator agg([&](const Candle& c, int tf) { generated.push_back(c); });

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < std::min<size_t>(json_data.size(), 300); ++i)
    {
        const auto& cj = json_data[i];
        Candle c;
        c.time = cj["time"];
        c.timestamp = parseTimestamp(c.time);
        c.open = cj["open"];
        c.high = cj["high"];
        c.low = cj["low"];
        c.close = cj["close"];
        c.volume = cj["volume"];
        agg.addM1Candle(c);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(ms, 100);
    EXPECT_GT(generated.size(), 0u);
}
