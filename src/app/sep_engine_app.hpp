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
 * Combines the functionality of OandaTraderApp and QuantumTrackerApp
 * Supports multiple operating modes via command-line flags
 */
class SepEngineApp {
public:
    enum class Mode {
        LIVE,           // Live trading mode (original OandaTraderApp)
        HISTORICAL_SIM, // Historical simulation mode (original QuantumTrackerApp historical)
        FILE_SIM        // File simulation mode (original QuantumTrackerApp file)
    };

    /**
     * Constructor
     * @param mode Operating mode of the application
     * @param headless Whether to run in headless mode (no GUI)
     */
    explicit SepEngineApp(Mode mode = Mode::LIVE, bool headless = true);
    
    /**
     * Constructor for simulation modes with specific parameters
     * @param mode Operating mode
     * @param simulate_start_time Start time for simulation (ISO format)
     * @param simulation_duration_hours Duration in hours for simulation
     */
    SepEngineApp(Mode mode, const std::string& simulate_start_time, int simulation_duration_hours);
    
    ~SepEngineApp();

    // Core lifecycle
    bool initialize();
    void run();
    void runHeadlessService();
    void shutdown();
    
    // Error handling
    const std::string& getLastError() const { return last_error_; }
    
    // Mode accessors
    Mode getMode() const { return mode_; }
    bool isHeadless() const { return headless_mode_; }

private:
    // Mode-specific initialization
    bool initializeLiveMode();
    bool initializeHistoricalSimMode();
    bool initializeFileSimMode();
    
    // Mode-specific execution
    void runLiveMode();
    void runHistoricalSimMode();
    void runFileSimMode();
    
    // OANDA integration (for live and historical modes)
    void connectToOanda();
    void refreshAccountInfo();
    void refreshPositions();
    void refreshOrderHistory();
    
    // Common functionality
    void initializeQuantumBridge();
    void initializeCuda();
    
    // Members common to all modes
    Mode mode_;
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
    
    // Simulation specific (QuantumTrackerApp members)
    std::string simulation_start_time_;
    int simulation_duration_hours_ = 0;
    bool historical_sim_mode_ = false;
    bool file_sim_mode_ = false;
    
    std::unique_ptr<sep::core::UnifiedDataManager> unified_data_manager_;
    std::unique_ptr<sep::connectors::TickDataManager> tick_data_manager_;
    std::unique_ptr<sep::trading::MarketModelCache> cache_;
    
    // Simulation control
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> is_running_{false};
};

} // namespace sep::apps