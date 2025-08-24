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
              << "  --headless              Run in headless mode (no GUI)\n"
              << "  --help                  Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    bool headless = true;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--headless") {
            headless = true;
        } else {
            std::cerr << "[WARNING] Unknown argument: " << arg << std::endl;
        }
    }

    try {
        // Create the unified application
        auto app = std::make_unique<sep::apps::SepEngineApp>(headless);

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