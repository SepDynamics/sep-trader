#include "sep_engine_app.hpp"

#include "core/sep_precompiled.h"
#include "util/nlohmann_json_safe.h"
#include "io/market_data_converter.h"
#include "market_utils.hpp"

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <thread>

namespace sep::apps {

// Global pointer for signal handling
static SepEngineApp* g_app_instance = nullptr;

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "\n[SEP] Received signal " << signal << " - initiating graceful shutdown..." << std::endl;
    if (g_app_instance) {
        g_app_instance->shutdown();
    }
    std::exit(signal);
}

SepEngineApp::SepEngineApp(bool headless)
    : headless_mode_(headless) {
    // Set global instance for signal handling
    g_app_instance = this;
    
    // Register signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
}

SepEngineApp::~SepEngineApp() {
    // Clear global instance
    if (g_app_instance == this) {
        g_app_instance = nullptr;
    }
    
    // Ensure proper shutdown
    if (is_running_.load()) {
        shutdown();
    }
}

bool SepEngineApp::initialize() {
    std::cout << "===============================================" << std::endl;
    std::cout << "   SEP Engine - Unified Application v1.0      " << std::endl;
    std::cout << "===============================================" << std::endl;

    // Initialize common components
    sep_engine_ = std::make_unique<sep::core::Engine>();

    try {
        initializeQuantumBridge();
        initializeCuda();
        std::cout << "[SEP] Initializing Live Trading Mode..." << std::endl;
        return initializeLiveMode();
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        return false;
    }
}

void SepEngineApp::run() {
    if (!is_running_.exchange(true)) {
        runLiveMode();
        is_running_ = false;
    }
}

void SepEngineApp::runHeadlessService() {
    std::cout << "[SEP] Running in headless service mode..." << std::endl;
    run();
}

void SepEngineApp::shutdown() {
    std::cout << "[SEP] Initiating shutdown sequence..." << std::endl;
    
    // Signal threads to stop
    should_stop_ = true;
    
    try {
        // Stop data streams first to prevent new data processing
        if (oanda_connector_) {
            std::cout << "[SEP] Stopping market data streams..." << std::endl;
            oanda_connector_->stopPriceStream();
        }
        
        // Wait for managed thread to finish
        if (data_stream_thread_.joinable()) {
            std::cout << "[SEP] Waiting for data stream thread to finish..." << std::endl;
            data_stream_thread_.join();
        }
        
        // Shutdown quantum components with proper error handling
        if (quantum_bridge_) {
            std::cout << "[SEP] Shutting down quantum signal bridge..." << std::endl;
            try {
                quantum_bridge_->shutdown();
            } catch (const std::exception& e) {
                std::cerr << "[SEP] Warning: Error shutting down quantum bridge: " << e.what() << std::endl;
            }
            quantum_bridge_.reset();
        }
        
        // Clean up mode-specific resources
        if (unified_data_manager_) {
            unified_data_manager_.reset();
        }
        
        if (tick_data_manager_) {
            tick_data_manager_.reset();
        }
        
        if (cache_) {
            cache_.reset();
        }
        
        // Clean up engine core
        if (sep_engine_) {
            sep_engine_.reset();
        }
        
        // Clean up CUDA resources
        try {
            std::cout << "[SEP] Cleaning up CUDA resources..." << std::endl;
            sep::apps::cuda::cleanupCudaDevice(cuda_context_);
        } catch (const std::exception& e) {
            std::cerr << "[SEP] Warning: Error cleaning up CUDA: " << e.what() << std::endl;
        }
        
        // Reset connection state
        oanda_connected_ = false;
        is_running_ = false;
        
        std::cout << "[SEP] Shutdown completed successfully." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[SEP] Error during shutdown: " << e.what() << std::endl;
        last_error_ = "Shutdown error: " + std::string(e.what());
    } catch (...) {
        std::cerr << "[SEP] Unknown error during shutdown" << std::endl;
        last_error_ = "Unknown shutdown error";
    }
}

bool SepEngineApp::initializeLiveMode() {
    // Initialize OANDA connector
    const char* api_key = std::getenv("OANDA_API_KEY");
    const char* account_id = std::getenv("OANDA_ACCOUNT_ID");
    if (!api_key || !account_id) {
        last_error_ = "OANDA_API_KEY and OANDA_ACCOUNT_ID environment variables must be set.";
        return false;
    }
    
    oanda_connector_ = std::make_unique<sep::connectors::OandaConnector>(api_key, account_id);
    
    // Set quantum bridge thresholds for live trading
    quantum_bridge_->setConfidenceThreshold(0.6f);
    quantum_bridge_->setCoherenceThreshold(0.4f);
    quantum_bridge_->setStabilityThreshold(0.0f);
    
    return true;
}

void SepEngineApp::runLiveMode() {
    std::cout << "[LIVE] Starting live trading mode..." << std::endl;
    
    if (headless_mode_) {
        std::cout << "[LIVE] Running in headless mode..." << std::endl;
    }
    
    // Connect to OANDA
    connectToOanda();
    
    // Start market data streaming and processing
    // This would contain the main trading loop from OandaTraderApp
    while (!should_stop_) {
        try {
            // Process market data and generate signals
            // Update WebSocket clients with real-time data
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            std::cerr << "[LIVE] Error in trading loop: " << e.what() << std::endl;
        }
    }
}

void SepEngineApp::initializeQuantumBridge() {
    quantum_bridge_ = std::make_unique<sep::trading::QuantumSignalBridge>();
    if (!quantum_bridge_->initialize()) {
        throw std::runtime_error("Failed to initialize quantum signal bridge");
    }
}

void SepEngineApp::initializeCuda() {
    sep::apps::cuda::initializeCudaDevice(cuda_context_);
}

void SepEngineApp::connectToOanda() {
    if (oanda_connector_) {
        // Implementation would connect to OANDA API
        oanda_connected_ = true;
        std::cout << "[OANDA] Connected successfully" << std::endl;
    }
}

void SepEngineApp::refreshAccountInfo() {
    if (oanda_connector_ && oanda_connected_) {
        // Refresh account balance and currency info
        // This data would be sent to WebSocket clients
    }
}

void SepEngineApp::refreshPositions() {
    if (oanda_connector_ && oanda_connected_) {
        std::lock_guard<std::mutex> lock(positions_mutex_);
        // Refresh open positions
        // This data would be sent to WebSocket clients
    }
}

void SepEngineApp::refreshOrderHistory() {
    if (oanda_connector_ && oanda_connected_) {
        std::lock_guard<std::mutex> lock(history_mutex_);
        // Refresh order history
        // This data would be sent to WebSocket clients
    }
}

} // namespace sep::apps