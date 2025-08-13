#include "nlohmann_json_safe.h"
// SEP Professional Training Coordinator Implementation
// Coordinates local CUDA training with remote trading deployment

#include "training_coordinator.hpp"

#include <curl/curl.h>
#include <spdlog/spdlog.h>

#include <algorithm>

#include "sep_precompiled.h"
#include "oanda_connector.h"
#include "weekly_cache_manager.hpp"
#include "quantum_pair_trainer.hpp"

// Restore array if corrupted
#ifdef array
#undef array
#endif

// Don't include standard_includes.h as it may cause issues
// #include "standard_includes.h"

using namespace sep::training;

TrainingCoordinator::TrainingCoordinator()
    : remote_connected_(false), sync_running_(false), live_tuning_active_(false)
{
    if (!initializeComponents())
    {
        throw std::runtime_error("Failed to initialize training coordinator");
    }

    loadTrainingResults();
}

TrainingCoordinator::~TrainingCoordinator()
{
    // Stop all running threads
    if (sync_running_)
    {
        sync_running_ = false;
        if (sync_thread_.joinable())
        {
            sync_thread_.join();
        }
    }

    if (live_tuning_active_)
    {
        stopLiveTuning();
    }
}

bool TrainingCoordinator::initializeComponents()
{
    try
    {
        // Initialize configuration manager
        config_manager_ = std::make_unique<config::DynamicConfigManager>();

        // Initialize cache manager
        cache_manager_ = std::make_unique<cache::WeeklyCacheManager>();

        // Set up OANDA data source provider for cache manager
        cache_manager_->setDataSourceProvider(
            [](const std::string& pair_symbol, std::chrono::system_clock::time_point from,
               std::chrono::system_clock::time_point to) -> std::vector<std::string> {
                const char* api_key = std::getenv("OANDA_API_KEY");
                const char* account_id = std::getenv("OANDA_ACCOUNT_ID");

                if (!api_key || !account_id)
                {
                    spdlog::warn("OANDA credentials not available for data fetching");
                    return {};
                }

                try
                {
                    auto oanda_connector = std::make_unique<sep::connectors::OandaConnector>(
                        api_key, account_id, true);
                    if (!oanda_connector->initialize())
                    {
                        spdlog::error("Failed to initialize OANDA connector for cache data");
                        return {};
                    }

                    // Convert time points to ISO 8601 strings
                    auto time_t_from = std::chrono::system_clock::to_time_t(from);
                    auto time_t_to = std::chrono::system_clock::to_time_t(to);

                    char from_str[32];
                    char to_str[32];
                    strftime(from_str, sizeof(from_str), "%Y-%m-%dT%H:%M:%S.000000000Z",
                             gmtime(&time_t_from));
                    strftime(to_str, sizeof(to_str), "%Y-%m-%dT%H:%M:%S.000000000Z",
                             gmtime(&time_t_to));

                    // Fetch historical data from OANDA
                    auto candles =
                        oanda_connector->getHistoricalData(pair_symbol, "M1", from_str, to_str);

                    // Convert candles to string format for cache storage
                    std::vector<std::string> result;
                    for (const auto& candle : candles)
                    {
                        nlohmann::json candle_json;
                        candle_json["timestamp"] = candle.timestamp;
                        candle_json["open"] = candle.open;
                        candle_json["high"] = candle.high;
                        candle_json["low"] = candle.low;
                        candle_json["close"] = candle.close;
                        candle_json["volume"] = candle.volume;
                        result.push_back(candle_json.dump());
                    }

                    spdlog::info("Fetched {} candles for {} from OANDA", result.size(),
                                 pair_symbol);
                    return result;
                }
                catch (const std::exception& e)
                {
                    spdlog::error("Error fetching OANDA data for cache: {}", e.what());
                    return {};
                }
            });

        // Convert candles to string format for cache storage
        std::vector<std::string> result;
        for (const auto& candle : candles)
        {
            nlohmann::json candle_json;
            candle_json["timestamp"] = candle.timestamp;
            candle_json["open"] = candle.open;
            candle_json["high"] = candle.high;
            candle_json["low"] = candle.low;
            candle_json["close"] = candle.close;
            candle_json["volume"] = candle.volume;
            result.push_back(candle_json.dump());
        }

        spdlog::info("Fetched {} candles for {} from OANDA", result.size(), pair_symbol);
        return result;
    }
    catch (const std::exception& e)
    {
        spdlog::error("Error fetching OANDA data for cache: {}", e.what());
        return {};
    }
});

// Initialize weekly data fetcher
data_fetcher_ = std::make_unique<WeeklyDataFetcher>();
DataFetchConfig fetch_config;
fetch_config.oanda_api_key = std::getenv("OANDA_API_KEY") ? std::getenv("OANDA_API_KEY") : "";
fetch_config.oanda_account_id =
    std::getenv("OANDA_ACCOUNT_ID") ? std::getenv("OANDA_ACCOUNT_ID") : "";
fetch_config.oanda_environment = "practice";
fetch_config.instruments = getStandardForexPairs();
fetch_config.granularities = getStandardGranularities();
fetch_config.history_days = 7;
fetch_config.compress_data = false;
fetch_config.parallel_fetchers = 2;
data_fetcher_->configure(fetch_config);

// Initialize remote synchronizer
remote_synchronizer_ = std::make_unique<RemoteSynchronizer>();

return true;
}
catch (const std::exception& e)
{
    std::cerr << "Failed to initialize components: " << e.what() << std::endl;
    return false;
}
}

bool TrainingCoordinator::trainPair(const std::string& pair, TrainingMode mode)
{
    std::cout << "🔧 Training " << pair << " in "
              << (mode == TrainingMode::QUICK ? "QUICK" : "FULL") << " mode..." << std::endl;

    try
    {
        auto result = executeCudaTraining(pair, mode);

        if (result.quality != PatternQuality::UNKNOWN)
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            training_results_[pair] = result;
            last_trained_[pair] = std::chrono::system_clock::now();

            saveTrainingResult(result);

            std::cout << "✅ " << pair << " training completed - " << result.accuracy
                      << "% accuracy" << std::endl;
            return true;
        }

        return false;
    }
    catch (const std::exception& e)
    {
        std::cerr << "❌ Training failed for " << pair << ": " << e.what() << std::endl;
        return false;
    }
}

TrainingResult TrainingCoordinator::executeCudaTraining(const std::string& pair, TrainingMode mode)
{
    TrainingResult result;
    result.pair = pair;
    result.trained_at = std::chrono::system_clock::now();

    // Real CUDA training implementation - replaced simulation stub
    try
    {
#ifdef __CUDACC__
        // Launch actual CUDA training kernels
        std::vector<float> training_data(1000, 1.0f);
        std::vector<float> results(1000, 0.0f);

        // Use real market data from OANDA - we have this available now
        sep::trading::QuantumTrainingConfig config;
        sep::trading::QuantumPairTrainer trainer(config);

        // Fetch real OANDA market data for training
        auto market_data = trainer.fetchTrainingData(pair, 24);  // 24 hours of real data

        if (!market_data.empty())
        {
            // Convert market data to training format
            std::vector<float> price_data;
            price_data.reserve(market_data.size());
            for (const auto& md : market_data)
            {
                price_data.push_back(static_cast<float>(md.mid));
            }

            // Launch actual CUDA training with real market data
            launch_quantum_training(price_data.data(), results.data(), price_data.size(), 10);
        }
        else
        {
            throw std::runtime_error("No market data available for training");
        }

        // Calculate real accuracy from CUDA results
        float total_accuracy = 0.0f;
        for (const auto& val : results)
        {
            total_accuracy += val;
        }
        result.accuracy = (total_accuracy / results.size()) * 100.0;

        // Set realistic bounds based on actual computation
        result.stability_score = std::min(0.95, std::max(0.5, result.accuracy / 100.0));
        result.coherence_score = std::min(0.90, std::max(0.4, result.accuracy / 120.0));
        result.entropy_score = std::min(0.85, std::max(0.3, result.accuracy / 150.0));

#else
        // CPU fallback - still better than pure simulation
        result.accuracy = 60.73;  // Use proven baseline
        result.stability_score = 0.75;
        result.coherence_score = 0.65;
        result.entropy_score = 0.55;
#endif
    }
    catch (const std::exception& e)
    {
        // Fallback to baseline if CUDA training fails
        result.accuracy = 60.73;  // Proven performance baseline
        result.stability_score = 0.75;
        result.coherence_score = 0.65;
        result.entropy_score = 0.55;
    }
    if (result.accuracy == 60.73 && result.stability_score == 0.75 &&
        result.coherence_score == 0.65 && result.entropy_score == 0.55)
    {
        throw std::runtime_error("Stub training values detected");
    }
    result.quality = assessPatternQuality(result.accuracy);
    result.model_hash = generateModelHash(result);

    // Set training parameters based on mode
    if (mode == TrainingMode::QUICK)
    {
        result.parameters["iterations"] = 100;
        result.parameters["batch_size"] = 512;
    }
    else
    {
        result.parameters["iterations"] = 1000;
        result.parameters["batch_size"] = 1024;
    }

    return result;
}

bool TrainingCoordinator::trainAllPairs(TrainingMode mode)
{
    std::vector<std::string> pairs = {"EUR_USD", "GBP_USD", "USD_JPY",
                                      "AUD_USD", "USD_CHF", "USD_CAD"};

    std::cout << "🔧 Training " << pairs.size() << " pairs..." << std::endl;

    bool all_success = true;
    for (size_t i = 0; i < pairs.size(); ++i)
    {
        std::cout << "[" << (i + 1) << "/" << pairs.size() << "] ";
        if (!trainPair(pairs[i], mode))
        {
            all_success = false;
        }
    }

    return all_success;
}

TrainingResult TrainingCoordinator::getTrainingResult(const std::string& pair) const
{
    std::lock_guard<std::mutex> lock(results_mutex_);
    auto it = training_results_.find(pair);
    if (it != training_results_.end())
    {
        return it->second;
    }

    // Return empty result
    TrainingResult empty;
    empty.pair = pair;
    empty.quality = PatternQuality::UNKNOWN;
    return empty;
}

std::map<std::string, std::string> TrainingCoordinator::getSystemStatus() const
{
    std::map<std::string, std::string> status;

    status["status"] = "ready";
    status["training_pairs"] = std::to_string(training_results_.size());
    status["remote_connected"] = remote_connected_ ? "true" : "false";
    status["live_tuning"] = live_tuning_active_ ? "active" : "inactive";

    return status;
}

bool TrainingCoordinator::configureRemoteTrader(const RemoteTraderConfig& config)
{
    remote_config_ = config;

    std::string scheme = config.ssl_enabled ? "https://" : "http://";
    std::string url = scheme + config.host + ":" + std::to_string(config.port) + "/api/status";

    CURL* curl = curl_easy_init();
    if (!curl)
    {
        spdlog::error("CURL initialization failed");
        remote_connected_ = false;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);

    struct curl_slist* headers = nullptr;
    if (!config.auth_token.empty())
    {
        std::string auth = "Authorization: Bearer " + config.auth_token;
        headers = curl_slist_append(headers, auth.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    if (res == CURLE_OK)
    {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    }
    else
    {
        spdlog::error("Remote trader connection failed: {}", curl_easy_strerror(res));
    }

    if (headers)
    {
        curl_slist_free_all(headers);
    }
    curl_easy_cleanup(curl);

    if (res == CURLE_OK && http_code == 200)
    {
        remote_connected_ = true;
        spdlog::info("Remote trader connection configured");
        return true;
    }

    remote_connected_ = false;
    spdlog::error("Remote trader connection test failed with HTTP {}", http_code);
    return false;
}

bool TrainingCoordinator::syncPatternsToRemote()
{
    if (!remote_connected_)
    {
        spdlog::warn("Remote trader not connected");
        return false;
    }

    spdlog::info("Syncing patterns to remote trader...");
    if (!remote_synchronizer_ || !remote_synchronizer_->sync())
    {
        spdlog::error("Pattern synchronization failed");
        return false;
    }

    spdlog::info("Patterns synchronized successfully");
    return true;
}

bool TrainingCoordinator::fetchWeeklyDataForAll()
{
    if (!data_fetcher_)
    {
        spdlog::error("Weekly data fetcher not initialized");
        return false;
    }

    auto results = data_fetcher_->fetchAllInstruments();
    bool all_ok = true;
    for (const auto& r : results)
    {
        if (!r.success)
        {
            spdlog::error("Data fetch failed for {}: {}", r.instrument, r.error_message);
            all_ok = false;
        }
    }
    return all_ok;
}

PatternQuality TrainingCoordinator::assessPatternQuality(double accuracy) const
{
    if (accuracy >= 70.0) return PatternQuality::HIGH;
    if (accuracy >= 60.0) return PatternQuality::MEDIUM;
    return PatternQuality::LOW;
}

std::string TrainingCoordinator::generateModelHash(const TrainingResult& result) const
{
    std::ostringstream oss;
    oss << result.pair << "_" << result.accuracy << "_"
        << std::chrono::duration_cast<std::chrono::seconds>(result.trained_at.time_since_epoch())
               .count();
    return std::to_string(std::hash<std::string>{}(oss.str()));
}

bool TrainingCoordinator::saveTrainingResult(const TrainingResult& result)
{
    // Save to JSON file (simplified implementation)
    std::string filename = "cache/training_result_" + result.pair + ".json";
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << "{\n";
        file << "  \"pair\": \"" << result.pair << "\",\n";
        file << "  \"accuracy\": " << result.accuracy << ",\n";
        file << "  \"quality\": \"" << static_cast<int>(result.quality) << "\",\n";
        file << "  \"model_hash\": \"" << result.model_hash << "\"\n";
        file << "}\n";
        file.close();
        return true;
    }
    return false;
}

bool TrainingCoordinator::loadTrainingResults()
{
    // Load existing training results from cache
    // Simplified implementation for now
    return true;
}

std::vector<TrainingResult> TrainingCoordinator::getAllResults() const
{
    std::lock_guard<std::mutex> lock(results_mutex_);
    std::vector<TrainingResult> results;
    for (const auto& pair : training_results_)
    {
        results.push_back(pair.second);
    }
    return results;
}

bool TrainingCoordinator::startLiveTuning(const std::vector<std::string>& pairs)
{
    if (live_tuning_active_)
    {
        std::cout << "⚠️  Live tuning already active" << std::endl;
        return false;
    }

    live_tuning_active_ = true;
    std::cout << "🎯 Starting live tuning for " << pairs.size() << " pairs..." << std::endl;

    {
        std::lock_guard<std::mutex> lock(tuning_mutex_);
        for (const auto& p : pairs)
        {
            tuning_queue_.push(p);
        }
    }
    tuning_cv_.notify_all();

    // Start tuning thread
    tuning_thread_ = std::thread(&TrainingCoordinator::liveTuningThreadFunction, this);

    return true;
}

bool TrainingCoordinator::stopLiveTuning()
{
    if (!live_tuning_active_)
    {
        return false;
    }

    live_tuning_active_ = false;
    tuning_cv_.notify_all();

    if (tuning_thread_.joinable())
    {
        tuning_thread_.join();
    }

    std::cout << "⏹️  Live tuning stopped" << std::endl;
    return true;
}

void TrainingCoordinator::liveTuningThreadFunction()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(tuning_mutex_);
        tuning_cv_.wait(lock, [this] { return !tuning_queue_.empty() || !live_tuning_active_; });
        if (!live_tuning_active_ && tuning_queue_.empty())
        {
            break;
        }

        std::string pair = std::move(tuning_queue_.front());
        tuning_queue_.pop();
        lock.unlock();

        if (!performLiveTuning(pair))
        {
            spdlog::warn("Live tuning failed for {}", pair);
        }
    }
}

bool TrainingCoordinator::isLiveTuningActive() const { return live_tuning_active_; }

bool TrainingCoordinator::isRemoteTraderConnected() const { return remote_connected_; }

bool TrainingCoordinator::fetchWeeklyDataForPair(const std::string& pair)
{
    if (!data_fetcher_)
    {
        spdlog::error("Weekly data fetcher not initialized");
        return false;
    }
    auto result = data_fetcher_->fetchInstrument(pair);
    if (!result.success)
    {
        spdlog::error("Weekly data fetch failed for {}: {}", pair, result.error_message);
        return false;
    }
    spdlog::info("Weekly data fetched for {}", pair);
    return true;
}

bool TrainingCoordinator::syncParametersFromRemote()
{
    if (!remote_connected_)
    {
        spdlog::warn("Remote trader not connected");
        return false;
    }

    spdlog::info("Syncing parameters from remote trader...");
    if (!remote_synchronizer_ || !remote_synchronizer_->sync())
    {
        spdlog::error("Parameter synchronization failed");
        return false;
    }
    spdlog::info("Parameters synchronized successfully");
    return true;
}

bool TrainingCoordinator::performLiveTuning(const std::string& pair)
{
    return trainPair(pair, TrainingMode::LIVE_TUNE);
}
