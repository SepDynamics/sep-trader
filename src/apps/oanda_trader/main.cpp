// Project headers - include precompiled first
#include "common/sep_precompiled.h"

// Core system headers
#include <csignal>
#include <cstdlib>
#include <exception>

// App headers
#include "apps/oanda_trader/oanda_trader_app.hpp"
#include "apps/oanda_trader/quantum_tracker_app.hpp"

// SEP types
using sep::apps::OandaTraderApp;
using sep::apps::QuantumTrackerApp;

// Declare signal handling functions and constants
extern "C" {
    typedef void (*sighandler_t)(int);
    sighandler_t signal(int signum, sighandler_t handler);
}
#define SIGINT 2
#define SIGTERM 15

// Global app instances for signal handling
static sep::apps::OandaTraderApp* g_app = nullptr;
static sep::apps::QuantumTrackerApp* g_tracker_app = nullptr;

// Signal handler for graceful shutdown
void signalHandler(int sig) {
    cout << "\n[Main] Received signal " << sig << ", shutting down gracefully..." << endl;
    
    if (g_app) {
        g_app->shutdown();
    }
    if (g_tracker_app) {
        g_tracker_app->shutdown();
    }
    ::std::exit(0);
}

int main(int argc, char* argv[]) {
    // Install signal handlers
    (void)signal(SIGINT, signalHandler);
    (void)signal(SIGTERM, signalHandler);

    string mode("trader");
    
    for (int i = 1; i < argc; ++i) {
        string arg(argv[i]);
        if (arg == "--mode" && i + 1 < argc) {
            mode = string(argv[++i]);
        }
    }

    if (mode == "tracker") {
        cout << "===============================================" << endl;
        cout << "   SEP Quantum Tracker - v1.0                " << endl;
        cout << "===============================================" << endl << endl;
        sep::apps::QuantumTrackerApp app;
        g_tracker_app = &app;
        if (!app.initialize()) {
            cerr << "[ERROR] Failed to initialize: " << app.getLastError() << endl;
            return 1;
        }
        app.run();
        app.shutdown();
    } else {
        cout << "=====================================" << endl;
        cout << "   SEP OANDA Trader - v1.0          " << endl;
        cout << "=====================================" << endl << endl;
        
        try {
            // Create OANDA trader app
            auto app = make_unique<sep::apps::OandaTraderApp>();
            g_app = app.get();
            
            // Initialize
            cout << "[Main] Initializing OANDA Trader..." << endl;
            if (!app->initialize()) {
                cerr << "[Main] Failed to initialize OANDA Trader: " 
                      << app->getLastError() << endl;
                return 1;
            }
            
            // Run main loop
            app->run();
            
            // Cleanup
            app->shutdown();
            
            return 0;
            
        } catch (const std::exception& e) {
            cerr << "[Main] Fatal error: " << e.what() << endl;
            return 1;
        } catch (...) {
            cerr << "[Main] Unknown fatal error" << endl;
            return 1;
        }
    }

    return 0;
}
