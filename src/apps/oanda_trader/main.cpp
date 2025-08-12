#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <string>

#include "apps/oanda_trader/oanda_trader_app.hpp"
#include "apps/oanda_trader/quantum_tracker_app.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    string mode("trader");

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