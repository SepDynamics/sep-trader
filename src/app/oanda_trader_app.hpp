#pragma once

#include <array>
#include "util/nlohmann_json_safe.h"

#include "app/cuda_types.cuh"
#include "app/forward_window_kernels.cuh"
#include "app/quantum_signal_bridge.hpp"
#include "app/tick_cuda_kernels.cuh"
#include "core/engine.h"
#include "core/forward_window_result.h"
#include "io/oanda_connector.h"
#include "util/managed_thread.hpp"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sep::apps {

class OandaTraderApp {
public:
    explicit OandaTraderApp(bool headless = true) : headless_mode_(headless) {}
    ~OandaTraderApp() = default;

    // Core lifecycle
    bool initialize();
    void run();
    void shutdown();
    
    // Error handling
    const std::string& getLastError() const { return last_error_; }

private:
    // OANDA integration
    void connectToOanda();
    void refreshAccountInfo();
    void refreshPositions();
    void refreshOrderHistory();
    
    // Members
    std::unique_ptr<sep::connectors::OandaConnector> oanda_connector_;
    std::unique_ptr<sep::core::Engine> sep_engine_;
    
    bool oanda_connected_ = false;
    std::string account_balance_ = "N/A";
    std::string account_currency_ = "USD";
    
    // Real-time data handling
    sep::util::ManagedThread data_stream_thread_;
    std::mutex market_data_mutex_;
    std::map<std::string, sep::connectors::MarketData> market_data_map_;
    std::deque<sep::connectors::MarketData> market_history_;
    std::mutex market_history_mutex_;
    std::unique_ptr<sep::trading::QuantumSignalBridge> quantum_bridge_;
    sep::trading::QuantumTradingSignal last_signal_;
    std::mutex signal_mutex_;
    std::vector<sep::quantum::bitspace::ForwardWindowResult> forward_window_results_;
    std::vector<nlohmann::json> open_positions_;
    std::mutex positions_mutex_;

    std::vector<nlohmann::json> order_history_;
    std::mutex history_mutex_;
    
    // Error handling
    std::string last_error_;
    sep::apps::cuda::CudaContext cuda_context_;
    
    // Runtime settings
    bool headless_mode_ = true;
};

} // namespace sep::apps
