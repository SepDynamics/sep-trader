#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#define SIGINT  2
#define SIGTERM 15

typedef void (*sighandler_t)(int);
sighandler_t signal(int signum, sighandler_t handler);

#include "apps/oanda_trader/oanda_trader_app.hpp"
#include "apps/oanda_trader/quantum_tracker_app.hpp"

namespace sep {
    namespace apps {
        class OandaTraderApp;
        class QuantumTrackerApp;
    }
}

// Global app instances for signal handling
static sep::apps::OandaTraderApp* g_app = nullptr;
static sep::apps::QuantumTrackerApp* g_tracker_app = nullptr;

// Signal handler for graceful shutdown
void signalHandler(int sig) {
    std::cout << "\n[Main] Received signal " << sig << ", shutting down gracefully..." << std::endl;
    
    if (g_app) {
        g_app->shutdown();
    }
    if (g_tracker_app) {
        g_tracker_app->shutdown();
    }
    exit(0);
}

int main(int argc, char* argv[]) {
    // Install signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::string mode("trader");
    
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--mode" && i + 1 < argc) {
            mode = std::string(argv[++i]);
        }
    }

    if (mode == "tracker") {
        std::cout << "===============================================" << std::endl;
        std::cout << "   SEP Quantum Tracker - v1.0                " << std::endl;
        std::cout << "===============================================" << std::endl << std::endl;
        sep::apps::QuantumTrackerApp app;
        g_tracker_app = &app;
        if (!app.initialize()) {
            std::cerr << "[ERROR] Failed to initialize: " << app.getLastError() << std::endl;
            return 1;
        }
        app.run();
        app.shutdown();
    } else {
        std::cout << "=====================================" << std::endl;
        std::cout << "   SEP OANDA Trader - v1.0          " << std::endl;
        std::cout << "=====================================" << std::endl << std::endl;
        
        try {
            // Create OANDA trader app
            auto app = std::make_unique<sep::apps::OandaTraderApp>();
            g_app = app.get();
            
            // Initialize
            std::cout << "[Main] Initializing OANDA Trader..." << std::endl;
            if (!app->initialize()) {
                std::cerr << "[Main] Failed to initialize OANDA Trader: " 
                      << app->getLastError() << std::endl;
                return 1;
            }
            
            // Run main loop
            app->run();
            
            // Cleanup
            app->shutdown();
            
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "[Main] Fatal error: " << e.what() << std::endl;
            return 1;
        } catch (...) {
            std::cerr << "[Main] Unknown fatal error" << std::endl;
            return 1;
        }
    }

    return 0;
}