#include <iostream>
#include "quantum_tracker_app.hpp"

int main(int argc, char* argv[]) {
    // Initialize quantum tracker in historical simulation mode
    sep::apps::QuantumTrackerApp app(true); // Historical simulation mode
    
    std::cout << "SEP Quantum Tracker v1.0.0 - Real-time Transition Tracking System" << std::endl;
    std::cout << "Initializing CUDA-accelerated Bit-Transition Harmonic Analysis..." << std::endl;
    
    // Initialize the tracker
    if (!app.initialize()) {
        std::cerr << "Failed to initialize quantum tracker: " << app.getLastError() << std::endl;
        return 1;
    }
    
    std::cout << "Quantum tracker initialized successfully" << std::endl;
    std::cout << "Running quantum analysis simulation..." << std::endl;
    
    // Run the simulation headless
    app.runHeadlessService();
    
    std::cout << "Quantum tracking simulation completed successfully" << std::endl;
    
    // Cleanup
    app.shutdown();
    
    return 0;
}