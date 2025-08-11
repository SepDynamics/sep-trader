#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "trading/quantum_pair_trainer.hpp"
#include "training/weekly_data_fetcher.hpp"

using namespace std::chrono;

// Helper to generate sample candle data
static void write_sample_candles(const std::string &path)
{
    nlohmann::json candles = nlohmann::json::array();
    auto now = system_clock::now();
    for (int i = 0; i < 10; ++i)
    {
        auto t = now - minutes(10 - i);
        std::time_t tt = system_clock::to_time_t(t);
        std::tm tm = *std::gmtime(&tt);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
        double base = 1.0 + i * 0.001;
        candles.push_back({{"time", oss.str()},
                           {"open", base},
                           {"high", base + 0.001},
                           {"low", base - 0.001},
                           {"close", base},
                           {"volume", 1000}});
    }
    std::ofstream out(path);
    out << candles.dump();
}

TEST(PipelineIntegration, FullDataTrainingCacheCycle)
{
    using sep::training::WeeklyDataFetcher;
    using sep::training::WeeklyFetcherConfig;

    // Ensure cache directory and sample data exist
    std::filesystem::create_directories("cache/oanda");
    write_sample_candles("cache/oanda/EUR_USD_M1.json");

    // Step 1: Data fetch benchmark
    WeeklyFetcherConfig fetch_cfg;
    fetch_cfg.history_days = 1;
    fetch_cfg.parallel_fetchers = 1;
    WeeklyDataFetcher fetcher(fetch_cfg);
    auto fetch_start = high_resolution_clock::now();
    auto fetch_result = fetcher.fetchInstrument("EUR_USD");
    auto fetch_ms = duration_cast<milliseconds>(high_resolution_clock::now() - fetch_start).count();
    EXPECT_GT(fetch_result.candles_fetched, 0);
    EXPECT_LT(fetch_ms, 1000);

    // Step 2: Training using cached data
    sep::trading::QuantumTrainingConfig config;
    config.training_window_hours = 1;
    config.stability_weight = 0.4;
    config.coherence_weight = 0.1;
    config.entropy_weight = 0.5;
    sep::trading::QuantumPairTrainer trainer(config);
    auto result = trainer.trainPair("EUR_USD");
    EXPECT_GT(result.high_confidence_accuracy, 0.0);
    EXPECT_LT(result.high_confidence_accuracy, 1.0);
    EXPECT_NEAR(result.overall_accuracy, 0.63, 1e-6);
    EXPECT_NE(result.high_confidence_accuracy, 0.6073);

    // GPU performance benchmark
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    ASSERT_GT(device_count, 0);
    const int N = 1 << 16;
    float *d_mem = nullptr;
    cudaMalloc(&d_mem, N * sizeof(float));
    auto gpu_start = high_resolution_clock::now();
    cudaMemset(d_mem, 0, N * sizeof(float));
    cudaDeviceSynchronize();
    auto gpu_ms = duration_cast<milliseconds>(high_resolution_clock::now() - gpu_start).count();
    EXPECT_LT(gpu_ms, 100);
    cudaFree(d_mem);

    // Step 3: Cache store validation
    std::string cache_key = "EUR_USD_test_cache";
    trainer.updateCache(cache_key, result);
    EXPECT_TRUE(trainer.isCacheValid(cache_key));
}
