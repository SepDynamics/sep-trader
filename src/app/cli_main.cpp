// SEP Professional Training CLI
// Advanced interface for CUDA training coordination and remote sync

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <chrono>
#include <thread>

#include "core/training_coordinator.hpp"
#include "core/cli_commands.hpp"
#include "core/status_display.hpp"

using namespace sep::training;

void printHeader() {
    std::cout << "\n";
    std::cout << "🚀 SEP Professional Training Coordinator v2.0\n";
    std::cout << "   CUDA-Accelerated Pattern Training & Remote Sync\n";
    std::cout << "================================================================\n\n";
}

void printUsage() {
    std::cout << "Usage: trader_cli [command] [options]\n\n";

    std::cout << "TRAINING COMMANDS:\n";
    std::cout << "  status                     - Show comprehensive system status\n";
    std::cout << "  train <pair>              - Train specific currency pair\n";
    std::cout << "  train-all [--quick]       - Train all pairs (quick mode optional)\n";
    std::cout << "  train-selected <pairs>    - Train selected pairs (comma-separated)\n";
    std::cout << "  retrain-failed            - Retrain all failed pairs\n\n";
    
    std::cout << "DATA MANAGEMENT:\n";
    std::cout << "  fetch-weekly [pair]       - Fetch weekly data (all pairs or specific)\n";
    std::cout << "  validate-cache            - Validate all cached data\n";
    std::cout << "  cleanup-cache             - Clean up old cache files\n\n";
    
    std::cout << "REMOTE INTEGRATION:\n";
    std::cout << "  configure-remote <ip>     - Configure remote trader connection\n";
    std::cout << "  sync-patterns             - Sync trained patterns to remote trader\n";
    std::cout << "  sync-parameters           - Sync optimized parameters from remote\n";
    std::cout << "  test-connection           - Test remote trader connectivity\n\n";
    
    std::cout << "LIVE TUNING:\n";
    std::cout << "  start-tuning <pairs>      - Start live parameter tuning\n";
    std::cout << "  stop-tuning               - Stop live parameter tuning\n";
    std::cout << "  tuning-status             - Show live tuning status\n\n";
    
    std::cout << "SYSTEM OPERATIONS:\n";
    std::cout << "  benchmark                 - Run CUDA training benchmark\n";
    std::cout << "  system-health             - Comprehensive system health check\n";
    std::cout << "  monitor [--duration=300]  - Real-time monitoring mode\n\n";
    
    std::cout << "Examples:\n";
    std::cout << "  trader_cli train EUR_USD\n";
    std::cout << "  trader_cli train-all --quick\n";
    std::cout << "  trader_cli configure-remote 100.85.55.105\n";
    std::cout << "  trader_cli start-tuning EUR_USD,GBP_USD,USD_JPY\n\n";
}

int main(int argc, char* argv[]) {
    printHeader();
    
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string command = argv[1];
    std::vector<std::string> args(argv + 2, argv + argc);
    
    try {
        TrainingCoordinator coordinator;
        CLICommands cli(coordinator);
        StatusDisplay display(coordinator);
        
        // Initialize system
        std::cout << "🔧 Initializing training coordinator...\n";
        auto system_status = coordinator.getSystemStatus();
        
        if (system_status["status"] != "ready") {
            std::cout << "❌ System initialization failed\n";
            std::cout << "Status: " << system_status["status"] << "\n";
            if (system_status.count("error")) {
                std::cout << "Error: " << system_status["error"] << "\n";
            }
            return 1;
        }
        
        std::cout << "✅ Training coordinator ready\n\n";
        
        // Execute command
        bool success = false;
        
        if (command == "status") {
            success = display.showSystemStatus();
            
        } else if (command == "train") {
            if (args.empty()) {
                std::cout << "❌ Error: Missing currency pair\n";
                std::cout << "Usage: trader_cli train <pair>\n";
                return 1;
            }
            success = cli.trainPair(args[0]);
            
        } else if (command == "train-all") {
            bool quick_mode = false;
            for (const auto& arg : args) {
                if (arg == "--quick") quick_mode = true;
            }
            success = cli.trainAllPairs(quick_mode);
            
        } else if (command == "train-selected") {
            if (args.empty()) {
                std::cout << "❌ Error: Missing currency pairs\n";
                std::cout << "Usage: trader_cli train-selected <pairs>\n";
                return 1;
            }
            success = cli.trainSelectedPairs(args[0]);
            
        } else if (command == "retrain-failed") {
            success = cli.retrainFailedPairs();
            
        } else if (command == "fetch-weekly") {
            std::string pair = args.empty() ? "" : args[0];
            success = cli.fetchWeeklyData(pair);
            
        } else if (command == "validate-cache") {
            success = cli.validateCache();
            
        } else if (command == "cleanup-cache") {
            success = cli.cleanupCache();
            
        } else if (command == "configure-remote") {
            if (args.empty()) {
                std::cout << "❌ Error: Missing remote IP address\n";
                std::cout << "Usage: trader_cli configure-remote <ip>\n";
                return 1;
            }
            success = cli.configureRemoteTrader(args[0]);
            
        } else if (command == "sync-patterns") {
            success = cli.syncPatternsToRemote();
            
        } else if (command == "sync-parameters") {
            success = cli.syncParametersFromRemote();
            
        } else if (command == "test-connection") {
            success = cli.testRemoteConnection();
            
        } else if (command == "start-tuning") {
            if (args.empty()) {
                std::cout << "❌ Error: Missing currency pairs\n";
                std::cout << "Usage: trader_cli start-tuning <pairs>\n";
                return 1;
            }
            success = cli.startLiveTuning(args[0]);
            
        } else if (command == "stop-tuning") {
            success = cli.stopLiveTuning();
            
        } else if (command == "tuning-status") {
            success = display.showTuningStatus();
            
        } else if (command == "benchmark") {
            success = cli.runBenchmark();
            
        } else if (command == "system-health") {
            success = display.showSystemHealth();
            
        } else if (command == "monitor") {
            int duration = 300; // 5 minutes default
            for (const auto& arg : args) {
                if (arg.substr(0, 11) == "--duration=") {
                    duration = std::stoi(arg.substr(11));
                }
            }
            success = display.startMonitoringMode(duration);
            
        } else {
            std::cout << "❌ Unknown command: " << command << "\n\n";
            printUsage();
            return 1;
        }
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Fatal error: " << e.what() << "\n";
        return 1;
    }
}
