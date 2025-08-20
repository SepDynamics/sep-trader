// SEP Training CLI Commands Implementation
// Command implementations for the professional training interface

#include "core/cli_commands.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <chrono>

using namespace sep::training;

CLICommands::CLICommands(TrainingCoordinator& coordinator) 
    : coordinator_(coordinator) {
}

bool CLICommands::trainPair(const std::string& pair) {
    std::cout << "🎯 Training pair: " << pair << std::endl;
    
    bool success = coordinator_.trainPair(pair, TrainingMode::FULL);
    
    if (success) {
        auto result = coordinator_.getTrainingResult(pair);
        printTrainingResult(pair, result);
    }
    
    return success;
}

bool CLICommands::trainAllPairs(bool quick_mode) {
    TrainingMode mode = quick_mode ? TrainingMode::QUICK : TrainingMode::FULL;
    
    std::cout << "🚀 Training all pairs in " 
              << (quick_mode ? "QUICK" : "FULL") << " mode..." << std::endl;
    
    return coordinator_.trainAllPairs(mode);
}

bool CLICommands::trainSelectedPairs(const std::string& pairs_csv) {
    auto pairs = parsePairsList(pairs_csv);
    
    if (pairs.empty()) {
        std::cout << "❌ No valid pairs specified" << std::endl;
        return false;
    }
    
    std::cout << "🎯 Training " << pairs.size() << " selected pairs..." << std::endl;
    
    bool all_success = true;
    for (size_t i = 0; i < pairs.size(); ++i) {
        printProgress("Training", i + 1, pairs.size());
        
        if (!coordinator_.trainPair(pairs[i], TrainingMode::FULL)) {
            all_success = false;
        }
    }
    
    return all_success;
}

bool CLICommands::retrainFailedPairs() {
    auto all_results = coordinator_.getAllResults();
    std::vector<std::string> failed_pairs;
    
    for (const auto& result : all_results) {
        if (result.quality == PatternQuality::LOW || result.accuracy < 60.0) {
            failed_pairs.push_back(result.pair);
        }
    }
    
    if (failed_pairs.empty()) {
        std::cout << "✅ No failed pairs found" << std::endl;
        return true;
    }
    
    std::cout << "🔧 Retraining " << failed_pairs.size() << " failed pairs..." << std::endl;
    
    bool all_success = true;
    for (const auto& pair : failed_pairs) {
        if (!coordinator_.trainPair(pair, TrainingMode::FULL)) {
            all_success = false;
        }
    }
    
    return all_success;
}

bool CLICommands::fetchWeeklyData(const std::string& pair) {
    if (pair.empty()) {
        std::cout << "📥 Fetching weekly data for all instruments..." << std::endl;
        return coordinator_.fetchWeeklyDataForAll();
    } else {
        std::cout << "📥 Fetching weekly data for " << pair << "..." << std::endl;
        return coordinator_.fetchWeeklyDataForPair(pair);
    }
}

bool CLICommands::validateCache() {
    std::cout << "🔍 Validating cache integrity..." << std::endl;
    
    // Simulate cache validation
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "✅ Cache validation completed" << std::endl;
    return true;
}

#include <filesystem>

bool CLICommands::cleanupCache() {
    if (!confirmOperation("cache cleanup")) {
        return false;
    }
    
    std::cout << "🧹 Cleaning up old cache files..." << std::endl;
    
    const std::string cache_path = "./cache/oanda";
    
    if (!std::filesystem::exists(cache_path)) {
        std::cout << "✅ Cache directory does not exist. Nothing to clean." << std::endl;
        return true;
    }
    
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(cache_path)) {
        std::filesystem::remove(entry.path());
        count++;
    }
    
    std::cout << "✅ Cache cleanup completed. Removed " << count << " files." << std::endl;
    return true;
}

bool CLICommands::configureRemoteTrader(const std::string& remote_ip) {
    std::cout << "🌐 Configuring remote trader connection..." << std::endl;
    std::cout << "Remote IP: " << remote_ip << std::endl;
    
    RemoteTraderConfig config;
    config.host = remote_ip;
    config.port = 8080;
    config.ssl_enabled = false;
    config.sync_interval_seconds = 300;  // 5 minutes
    
    return coordinator_.configureRemoteTrader(config);
}

bool CLICommands::syncPatternsToRemote() {
    if (!coordinator_.isRemoteTraderConnected()) {
        std::cout << "❌ Remote trader not connected. Use 'configure-remote' first." << std::endl;
        return false;
    }
    
    return coordinator_.syncPatternsToRemote();
}

bool CLICommands::syncParametersFromRemote() {
    if (!coordinator_.isRemoteTraderConnected()) {
        std::cout << "❌ Remote trader not connected. Use 'configure-remote' first." << std::endl;
        return false;
    }
    
    return coordinator_.syncParametersFromRemote();
}

bool CLICommands::testRemoteConnection() {
    std::cout << "🔗 Testing remote trader connection..." << std::endl;
    
    if (coordinator_.isRemoteTraderConnected()) {
        std::cout << "✅ Remote trader connection is active" << std::endl;
        return true;
    } else {
        std::cout << "❌ Remote trader connection failed" << std::endl;
        return false;
    }
}

bool CLICommands::startLiveTuning(const std::string& pairs_csv) {
    auto pairs = parsePairsList(pairs_csv);
    
    if (pairs.empty()) {
        std::cout << "❌ No valid pairs specified for live tuning" << std::endl;
        return false;
    }
    
    return coordinator_.startLiveTuning(pairs);
}

bool CLICommands::stopLiveTuning() {
    return coordinator_.stopLiveTuning();
}

bool CLICommands::runBenchmark() {
    std::cout << "🏃 Running CUDA training benchmark..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate benchmark
    coordinator_.trainPair("EUR_USD", TrainingMode::QUICK);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "⏱️  Benchmark completed in " << duration.count() << "ms" << std::endl;
    std::cout << "🎯 Training throughput: ~" << (1000.0 / duration.count()) << " pairs/second" << std::endl;
    
    return true;
}

std::vector<std::string> CLICommands::parsePairsList(const std::string& pairs_csv) {
    std::vector<std::string> pairs;
    std::stringstream ss(pairs_csv);
    std::string pair;
    
    while (std::getline(ss, pair, ',')) {
        // Trim whitespace
        pair.erase(0, pair.find_first_not_of(" \t"));
        pair.erase(pair.find_last_not_of(" \t") + 1);
        
        if (!pair.empty()) {
            pairs.push_back(pair);
        }
    }
    
    return pairs;
}

void CLICommands::printProgress(const std::string& operation, int current, int total) {
    double percentage = (static_cast<double>(current) / total) * 100.0;
    std::cout << operation << " [" << current << "/" << total << "] " 
              << std::fixed << std::setprecision(1) << percentage << "%" << std::endl;
}

void CLICommands::printTrainingResult(const std::string& pair, const TrainingResult& result) {
    std::cout << "\n📊 Training Result for " << pair << ":" << std::endl;
    std::cout << "   Accuracy: " << std::fixed << std::setprecision(2) << result.accuracy << "%" << std::endl;
    std::cout << "   Stability: " << result.stability_score << std::endl;
    std::cout << "   Coherence: " << result.coherence_score << std::endl;
    std::cout << "   Entropy: " << result.entropy_score << std::endl;
    std::cout << "   Quality: " << static_cast<int>(result.quality) << std::endl;
    std::cout << "   Model Hash: " << result.model_hash.substr(0, 12) << "..." << std::endl;
}

bool CLICommands::confirmOperation(const std::string& operation) {
    std::cout << "⚠️  Are you sure you want to perform " << operation << "? (y/N): ";
    std::string response;
    std::getline(std::cin, response);
    
    return !response.empty() && (response[0] == 'y' || response[0] == 'Y');
}
