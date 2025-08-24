#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "sep_engine_app.hpp"

using namespace std;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --mode <MODE>           Operating mode (default: live)\n"
              << "                          live          - Live trading mode\n"
              << "                          historical-sim - Historical simulation\n"
              << "                          file-sim      - File-based simulation\n"
              << "  --headless              Run in headless mode (no GUI)\n"
              << "  --start-time <TIME>     Start time for simulation (ISO format)\n"
              << "  --duration <HOURS>      Duration in hours for simulation\n"
              << "  --help                  Show this help message\n"
              << "\nExamples:\n"
              << "  " << program_name << " --mode live\n"
              << "  " << program_name << " --mode historical-sim --start-time 2024-01-01T00:00:00Z --duration 24\n"
              << "  " << program_name << " --mode file-sim --headless\n"
              << std::endl;
}

sep::apps::SepEngineApp::Mode parseMode(const std::string& mode_str) {
    if (mode_str == "live") {
        return sep::apps::SepEngineApp::Mode::LIVE;
    } else if (mode_str == "historical-sim") {
        return sep::apps::SepEngineApp::Mode::HISTORICAL_SIM;
    } else if (mode_str == "file-sim") {
        return sep::apps::SepEngineApp::Mode::FILE_SIM;
    } else if (mode_str == "tracker") {
        // Backward compatibility
        return sep::apps::SepEngineApp::Mode::HISTORICAL_SIM;
    } else if (mode_str == "trader") {
        // Backward compatibility
        return sep::apps::SepEngineApp::Mode::LIVE;
    }
    
    throw std::invalid_argument("Invalid mode: " + mode_str);
}

int main(int argc, char* argv[]) {
    sep::apps::SepEngineApp::Mode mode = sep::apps::SepEngineApp::Mode::LIVE;
    bool headless = true;
    std::string start_time;
    int duration_hours = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--mode" && i + 1 < argc) {
            try {
                mode = parseMode(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "[ERROR] " << e.what() << std::endl;
                std::cerr << "Use --help for usage information." << std::endl;
                return 1;
            }
        } else if (arg == "--headless") {
            headless = true;
        } else if (arg == "--start-time" && i + 1 < argc) {
            start_time = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            try {
                duration_hours = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Invalid duration: " << argv[i] << std::endl;
                return 1;
            }
        } else {
            std::cerr << "[WARNING] Unknown argument: " << arg << std::endl;
        }
    }

    try {
        // Create the unified application
        std::unique_ptr<sep::apps::SepEngineApp> app;
        
        if (!start_time.empty() && duration_hours > 0) {
            app = std::make_unique<sep::apps::SepEngineApp>(mode, start_time, duration_hours);
        } else {
            app = std::make_unique<sep::apps::SepEngineApp>(mode, headless);
        }

        // Initialize and run
        if (!app->initialize()) {
            std::cerr << "[ERROR] Failed to initialize: " << app->getLastError() << std::endl;
            return 1;
        }

        // Run the application
        if (headless) {
            app->runHeadlessService();
        } else {
            app->run();
        }

        // Clean shutdown
        app->shutdown();
        
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Application error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[FATAL] Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}