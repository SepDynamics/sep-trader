// SEP Main Entry Point - Professional Trading System
// Unified entry point that delegates to CLI interface

#include <cstdio>
#include <cstring>
#include <iostream>
#include <exception>

// Forward declaration of CLI main functionality
extern void printHeader();
extern void printUsage();
extern int handleCommand(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    try {
        // If no arguments provided, show header and usage
        if (argc <= 1) {
            printHeader();
            printUsage();
            return 0;
        }
        
        // Delegate to CLI command handler
        return handleCommand(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }
}
