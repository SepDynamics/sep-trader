#pragma once

// Include the precompiled header first
#include "util/stable_headers.h"

// Include necessary trading headers directly
#include "core/dynamic_pair_manager.hpp"
#include "core/quantum_pair_trainer.hpp"
#include "core/ticker_pattern_analyzer.hpp"
#include "core/trading_state.hpp"
#include "core/pair_manager.hpp"
#include "core/dynamic_config_manager.hpp"
#include "core/weekly_cache_manager.hpp"
#include "core/cache_health_monitor.hpp"
#include "core/cache_validator.hpp"
#include "core/trading_state.hpp"

// Forward declare needed classes and enums
namespace sep
{
    namespace core
    {
        // Forward declare only - no redefinition
        enum class SystemStatus;
        struct SystemHealth;
    }  // namespace core

    namespace trading
    {
        // Forward declare only - no redefinition
        struct TickerPatternAnalysis;
    }  // namespace trading
}  // namespace sep

namespace sep {
namespace cli {

struct CLICommand {
    std::string name;
    std::string description;
    std::function<int(const std::vector<std::string>&)> handler;
    std::vector<std::string> subcommands;
};

class TraderCLI {
public:
    TraderCLI();
    ~TraderCLI();

    int run(int argc, char* argv[]);
    void print_help() const;
    void print_version() const;

private:
    void register_commands();
    int execute_command(const std::string& command, const std::vector<std::string>& args);
    
    // Command handlers
    int handle_start(const std::vector<std::string>& args);
    int handle_stop(const std::vector<std::string>& args);
    int handle_status(const std::vector<std::string>& args);
    int handle_pairs(const std::vector<std::string>& args);
    int handle_cache(const std::vector<std::string>& args);
    int handle_config(const std::vector<std::string>& args);
    int handle_logs(const std::vector<std::string>& args);
    int handle_metrics(const std::vector<std::string>& args);
    int handle_data_import(const std::vector<std::string>& args);

    // New command handlers for trading/quantum functionality
    int handle_trading(const std::vector<std::string>& args);
    int handle_training(const std::vector<std::string>& args);
    int handle_analysis(const std::vector<std::string>& args);
    int handle_quantum_config(const std::vector<std::string>& args);
    int handle_quantum_monitor(const std::vector<std::string>& args);
    int handle_quantum_status(const std::vector<std::string>& args);
    int handle_quantum_list_pairs(const std::vector<std::string>& args);
    int handle_quantum_enable(const std::vector<std::string>& args);
    int handle_quantum_disable(const std::vector<std::string>& args);

    // Helper methods
    void print_status_table() const;
    void print_pairs_table() const;
    void print_cache_status() const;
    void print_recent_logs(int lines = 20) const;
    void print_performance_metrics() const;
    
    // Helper methods for quantum/trading
    void print_quantum_training_usage() const;
    void print_quantum_analysis_usage() const;
    void print_quantum_config_usage() const;
    void print_quantum_monitor_usage() const;
    void print_quantum_status_table() const;
    void print_quantum_pair_status(const std::string& pair) const;
    void print_quantum_list_pairs(const std::vector<std::string>& args) const;
    void print_quantum_config_details() const;


    std::map<std::string, CLICommand> commands_;
    bool verbose_;
    std::string config_path_;

    // Members for quantum training and analysis
    std::unique_ptr<sep::trading::QuantumPairTrainer> trainer_;
    std::unique_ptr<sep::trading::TickerPatternAnalyzer> analyzer_;
    std::unique_ptr<sep::trading::DynamicPairManager> dynamic_pair_manager_; // Renamed to avoid conflict with core::PairManager
    sep::cache::ValidationResponse validation_response_;
};

} // namespace cli
} // namespace sep
