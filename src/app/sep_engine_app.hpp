#pragma once

#include <array>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include "util/nlohmann_json_safe.h"
#include "app/cuda_types.cuh"
#include "app/forward_window_kernels.cuh"
#include "app/quantum_signal_bridge.hpp"
#include "app/tick_cuda_kernels.cuh"
#include "app/candle_types.h"
#include "core/engine.h"
#include "core/forward_window_result.h"
#include "core/unified_data_manager.hpp"
#include "io/oanda_connector.h"
#include "util/managed_thread.hpp"
#include "tick_data_manager.hpp"
#include "market_model_cache.hpp"

namespace sep::apps {

/**
 * Unified SEP Engine Application
 * Combines core trading and analysis components
 * Simplified to operate solely in live trading mode
 */
class SepEngineApp {
public:
    /**
     * Constructor
     * @param headless Whether to run in headless mode (no GUI)
     */
    explicit SepEngineApp(bool headless = true);
    
    ~SepEngineApp();

    // Core lifecycle
    bool initialize();
    void run();
    void runHeadlessService();
    void shutdown();
    
    // Error handling
    const std::string& getLastError() const { return last_error_; }
    
    bool isHeadless() const { return headless_mode_; }

private:
    // Initialization and execution
    bool initializeLiveMode();
    void runLiveMode();
    
    // OANDA integration (for live and historical modes)
    void connectToOanda();
    void refreshAccountInfo();
    void refreshPositions();
    void refreshOrderHistory();
    
    // Common functionality
    void initializeQuantumBridge();
    void initializeCuda();
    
    // Members
    bool headless_mode_;
    std::string last_error_;

    // Quantum processing components
    std::unique_ptr<sep::trading::QuantumSignalBridge> quantum_bridge_;
    sep::trading::QuantumTradingSignal last_signal_;
    std::mutex signal_mutex_;
    
    // SEP engine core
    std::unique_ptr<sep::core::Engine> sep_engine_;
    
    // CUDA context
    sep::apps::cuda::CudaContext cuda_context_;
    
    // Live trading specific (OandaTraderApp members)
    std::unique_ptr<sep::connectors::OandaConnector> oanda_connector_;
    bool oanda_connected_ = false;
    std::string account_balance_ = "N/A";
    std::string account_currency_ = "USD";
    
    sep::util::ManagedThread data_stream_thread_;
    std::mutex market_data_mutex_;
    std::map<std::string, sep::connectors::MarketData> market_data_map_;
    std::deque<sep::connectors::MarketData> market_history_;
    std::mutex market_history_mutex_;
    std::vector<sep::quantum::bitspace::ForwardWindowResult> forward_window_results_;
    std::vector<nlohmann::json> open_positions_;
    std::mutex positions_mutex_;
    std::vector<nlohmann::json> order_history_;
    std::mutex history_mutex_;
    
    std::unique_ptr<sep::trading::UnifiedDataManager> unified_data_manager_;
    std::unique_ptr<sep::apps::TickDataManager> tick_data_manager_;
    std::unique_ptr<sep::apps::MarketModelCache> cache_;
    
    // Execution control
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> is_running_{false};
};

} // namespace sep::apps