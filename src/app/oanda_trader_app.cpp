#include "util/nlohmann_json_safe.h"
#include "oanda_trader_app.hpp"

#include "core/sep_precompiled.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>

#include "io/market_data_converter.h"
#include "core/engine.h"

namespace sep::apps {

bool OandaTraderApp::initialize() {
    // Initialize OANDA connector
    const char* api_key = std::getenv("OANDA_API_KEY");
    const char* account_id = std::getenv("OANDA_ACCOUNT_ID");
    if (!api_key || !account_id) {
        last_error_ = "OANDA_API_KEY and OANDA_ACCOUNT_ID environment variables must be set.";
        return false;
    }
    oanda_connector_ = std::make_unique<sep::connectors::OandaConnector>(api_key, account_id);
    
    // Initialize SEP engine
    sep_engine_ = std::make_unique<sep::core::Engine>();

    // Initialize quantum bridge for signal processing
    quantum_bridge_ = std::make_unique<sep::trading::QuantumSignalBridge>();
    if (!quantum_bridge_->initialize()) {
        last_error_ = "Failed to initialize quantum signal bridge";
        return false;
    }

    // Set default thresholds based on tracker configuration
    quantum_bridge_->setConfidenceThreshold(0.6f);
    quantum_bridge_->setCoherenceThreshold(0.4f);
    quantum_bridge_->setStabilityThreshold(0.0f);

    return true;
}

void OandaTraderApp::run() {
    sep::apps::cuda::initializeCudaDevice(cuda_context_);

    // Headless mode - run without GUI
    std::cout << "Running in headless mode..." << std::endl;
    
    // Run processing loop without GUI
    bool running = true;
    while (running) {
        // Perform forward-looking window calculations
        if (oanda_connected_ && !market_history_.empty()) {
            std::vector<sep::apps::cuda::TickData> ticks;
            {
                std::lock_guard<std::mutex> lock(market_history_mutex_);
                ticks.reserve(market_history_.size());
                for (const auto& md : market_history_) {
                    ticks.push_back({md.mid, md.bid, md.ask, md.timestamp, md.volume});
                }
            }
            const uint64_t window_size_ns = 24ULL * 3600ULL * 1000000000ULL; // 24 hours
            sep::apps::cuda::calculateForwardWindowsCuda(cuda_context_, ticks, forward_window_results_, window_size_ns);
        }
        
        // Sleep briefly to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Simple exit condition for headless mode (could be improved)
        static int iterations = 0;
        if (++iterations > 1000) running = false;
    }
}

void OandaTraderApp::connectToOanda() {
    if (!oanda_connector_) {
        std::cerr << "[OANDA] Connector not initialized" << std::endl;
        return;
    }
    
    // Check for environment variables
    const char* api_key = std::getenv("OANDA_API_KEY");
    const char* account_id = std::getenv("OANDA_ACCOUNT_ID");
    
    if (!api_key || !account_id) {
        std::cerr << "[OANDA] Missing environment variables. Set OANDA_API_KEY and OANDA_ACCOUNT_ID" << std::endl;
        oanda_connected_ = false;
        return;
    }
    
    std::cout << "[OANDA] Attempting to connect..." << std::endl;
    
    // Initialize the connector
    if (!oanda_connector_->initialize()) {
        std::cerr << "[OANDA] Failed to initialize connector: " << oanda_connector_->getLastError() << std::endl;
        oanda_connected_ = false;
        return;
    }
    
    oanda_connected_ = true;
    std::cout << "[OANDA] Successfully connected!" << std::endl;
    refreshAccountInfo();
    
    // Set the price callback
    oanda_connector_->setPriceCallback([this](const sep::connectors::MarketData& data) {
        {
            std::lock_guard<std::mutex> lock(market_data_mutex_);
            market_data_map_[data.instrument] = data;
        }

        {
            std::lock_guard<std::mutex> lock(market_history_mutex_);
            market_history_.push_back(data);
            if (market_history_.size() > 256) {
                market_history_.pop_front();
            }
        }

        if (quantum_bridge_) {
            std::vector<sep::connectors::MarketData> history_copy;
            {
                std::lock_guard<std::mutex> lock(market_history_mutex_);
                history_copy.assign(market_history_.begin(), market_history_.end());
            }

            auto signal = quantum_bridge_->analyzeMarketData(data, history_copy, forward_window_results_);
            {
                std::lock_guard<std::mutex> lock(signal_mutex_);
                last_signal_ = signal;
            }
            if (signal.should_execute) {
                std::cout << "[Signal] " << (signal.action == sep::trading::QuantumTradingSignal::BUY ? "BUY" : "SELL")
                          << " confidence:" << signal.identifiers.confidence << " size:" << signal.suggested_position_size << std::endl;
            }
        }
    });

    // Start the price stream using managed thread
    try {
        data_stream_thread_.start([this]() {
            std::cout << "[OANDA] Starting price stream for EUR_USD..." << std::endl;
            if (!oanda_connector_->startPriceStream({"EUR_USD"})) {
                std::cerr << "[OANDA] Failed to start price stream: "
                          << oanda_connector_->getLastError() << std::endl;
            }
        });
    } catch (const std::exception& e) {
        std::cerr << "[OANDA] Exception starting price stream: " << e.what() << std::endl;
    }
}

void OandaTraderApp::refreshAccountInfo() {
    if (!oanda_connected_ || !oanda_connector_) {
        account_balance_ = "N/A";
        account_currency_ = "N/A";
        return;
    }
    
    try {
        auto account_json = oanda_connector_->getAccountInfo();
        if (account_json.contains("account")) {
            account_balance_ = account_json["account"]["balance"];
            account_currency_ = account_json["account"]["currency"];
            
            std::cout << "[Account] Balance: " << account_balance_
                      << " " << account_currency_ << std::endl;
        } else if (account_json.contains("error")) {
            std::cerr << "[Account] Error: " << account_json["error"] << std::endl;
            account_balance_ = "Error";
            account_currency_ = "N/A";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Account] Exception: " << e.what() << std::endl;
        account_balance_ = "Error";
        account_currency_ = "N/A";
    }
}

void OandaTraderApp::refreshPositions() {
    if (!oanda_connected_) return;
    
    auto positions_json = oanda_connector_->getOpenPositions();
    if (positions_json.contains("positions")) {
        std::lock_guard<std::mutex> lock(positions_mutex_);
        open_positions_ = positions_json["positions"].get<std::vector<nlohmann::json>>();
    }
}

void OandaTraderApp::refreshOrderHistory() {
    if (!oanda_connected_) return;
    
    auto orders_json = oanda_connector_->getOrders();
    if (orders_json.contains("orders")) {
        std::lock_guard<std::mutex> lock(history_mutex_);
        order_history_ = orders_json["orders"].get<std::vector<nlohmann::json>>();
    }
}

void OandaTraderApp::shutdown() {
    sep::apps::cuda::cleanupCudaDevice(cuda_context_);
    if (oanda_connector_) {
        oanda_connector_->stopPriceStream();
    }
    data_stream_thread_.join();
    if (quantum_bridge_) {
        quantum_bridge_->shutdown();
    }
}

} // namespace sep::apps