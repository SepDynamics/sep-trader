#pragma once

#include "core/sep_precompiled.h"
#include "io/oanda_connector.h"
#include "core/quantum_pair_trainer.hpp"

namespace sep::backtesting
{
    /**
     * Backtesting Configuration
     * Professional backtesting parameters and settings
     */
    struct BacktestConfig
    {
        // Time range configuration
        std::string start_date = "2023-01-01";           // ISO format YYYY-MM-DD
        std::string end_date = "2024-12-31";             // ISO format YYYY-MM-DD
        size_t warmup_period_days = 30;                  // Model warmup period
        
        // Trading parameters
        double initial_capital = 10000.0;                // Starting capital
        double risk_per_trade = 0.02;                    // 2% risk per trade
        double max_daily_drawdown = 0.05;                // 5% max daily drawdown
        double max_position_size = 0.1;                  // 10% max position size
        
        // Execution settings
        double spread_cost_pips = 1.5;                   // Average spread cost
        double commission_per_lot = 3.5;                 // Commission per standard lot
        double slippage_pips = 0.5;                      // Average slippage
        size_t max_concurrent_trades = 5;                // Maximum open positions
        
        // Model evaluation parameters
        double accuracy_threshold = 0.55;                // Minimum acceptable accuracy
        double sharpe_ratio_threshold = 1.0;             // Minimum Sharpe ratio
        double max_consecutive_losses = 10;              // Max losing streak
        double profit_factor_threshold = 1.3;            // Minimum profit factor
        
        // Advanced backtesting features
        bool enable_walk_forward = true;                 // Walk-forward analysis
        size_t walk_forward_periods = 12;                // Number of WF periods
        bool enable_monte_carlo = true;                  // Monte Carlo simulation
        size_t monte_carlo_runs = 1000;                  // MC simulation runs
        bool enable_stress_testing = true;              // Stress test scenarios
        
        // Reporting configuration
        bool generate_detailed_report = true;           // Comprehensive reporting
        bool export_trade_log = true;                   // Export individual trades
        bool create_equity_curve = true;                // Generate equity curve
        std::string output_directory = "output/backtest/"; // Report output path
    };

    /**
     * Trade Record
     * Individual trade execution record
     */
    struct TradeRecord
    {
        size_t trade_id = 0;
        std::string pair_symbol;
        std::string trade_type; // "BUY" or "SELL"
        double entry_price = 0.0;
        double exit_price = 0.0;
        double position_size = 0.0; // In lots
        
        std::string entry_timestamp;
        std::string exit_timestamp;
        size_t duration_minutes = 0;
        
        double gross_pnl = 0.0;    // Before costs
        double net_pnl = 0.0;      // After costs
        double spread_cost = 0.0;
        double commission = 0.0;
        double slippage_cost = 0.0;
        
        double confidence_score = 0.0;  // Model confidence
        std::string entry_reason;       // Why trade was taken
        std::string exit_reason;        // Why trade was closed
        
        bool is_winning_trade = false;
        double max_favorable_excursion = 0.0; // MFE
        double max_adverse_excursion = 0.0;   // MAE
        double equity_before_trade = 0.0;
        double equity_after_trade = 0.0;
    };

    /**
     * Backtesting Results
     * Comprehensive backtesting performance metrics
     */
    struct BacktestResult
    {
        // Basic performance metrics
        double total_return_percentage = 0.0;
        double annualized_return = 0.0;
        double max_drawdown = 0.0;
        double sharpe_ratio = 0.0;
        double sortino_ratio = 0.0;
        double profit_factor = 0.0;
        
        // Trade statistics
        size_t total_trades = 0;
        size_t winning_trades = 0;
        size_t losing_trades = 0;
        double win_rate = 0.0;
        double average_win = 0.0;
        double average_loss = 0.0;
        double largest_win = 0.0;
        double largest_loss = 0.0;
        
        // Risk metrics
        double maximum_consecutive_wins = 0;
        double maximum_consecutive_losses = 0;
        double average_trade_duration_hours = 0.0;
        double volatility_of_returns = 0.0;
        double value_at_risk_95 = 0.0;        // 95% VaR
        double conditional_var_95 = 0.0;      // Expected Shortfall
        
        // Advanced metrics
        double calmar_ratio = 0.0;             // Return/Max Drawdown
        double sterling_ratio = 0.0;           // Risk-adjusted performance
        double ulcer_index = 0.0;             // Drawdown volatility
        double pain_index = 0.0;              // Drawdown severity
        
        // Model accuracy metrics
        double prediction_accuracy = 0.0;      // Overall accuracy
        double high_confidence_accuracy = 0.0; // High confidence subset
        double precision = 0.0;                // True positive rate
        double recall = 0.0;                   // Sensitivity
        double f1_score = 0.0;                 // Harmonic mean of precision/recall
        
        // Trading costs impact
        double total_spread_costs = 0.0;
        double total_commissions = 0.0;
        double total_slippage_costs = 0.0;
        double cost_percentage_of_returns = 0.0;
        
        // Temporal analysis
        std::map<std::string, double> monthly_returns;     // Month -> Return
        std::map<std::string, double> quarterly_returns;   // Quarter -> Return
        std::map<int, double> yearly_returns;              // Year -> Return
        std::map<int, double> hourly_performance;          // Hour -> Avg Return
        std::map<int, double> daily_performance;           // Day of week -> Return
        
        // Walk-forward analysis results
        std::vector<double> walk_forward_returns;
        double walk_forward_stability = 0.0;
        double walk_forward_consistency = 0.0;
        
        // Monte Carlo results
        std::vector<double> monte_carlo_returns;
        double monte_carlo_mean_return = 0.0;
        double monte_carlo_std_return = 0.0;
        double monte_carlo_percentile_5 = 0.0;   // 5th percentile
        double monte_carlo_percentile_95 = 0.0;  // 95th percentile
        
        // Trade records and equity curve
        std::vector<TradeRecord> trade_history;
        std::vector<std::pair<std::string, double>> equity_curve; // Timestamp, Equity
        
        // Validation and status
        bool passed_accuracy_threshold = false;
        bool passed_sharpe_threshold = false;
        bool passed_profit_factor_threshold = false;
        std::string overall_grade;              // A, B, C, D, F
        std::vector<std::string> warnings;
        std::vector<std::string> recommendations;
    };

    /**
     * Production-Grade Backtesting Engine
     * Professional backtesting system with comprehensive analysis
     */
    class BacktestingEngine
    {
    public:
        explicit BacktestingEngine(const BacktestConfig& config = {});
        ~BacktestingEngine();

        // Primary backtesting interface
        BacktestResult runBacktest(
            const std::string& pair_symbol,
            std::unique_ptr<sep::trading::QuantumPairTrainer> model);
        
        // Multi-pair backtesting
        std::map<std::string, BacktestResult> runMultiPairBacktest(
            const std::vector<std::string>& pair_symbols,
            std::unique_ptr<sep::trading::QuantumPairTrainer> model);
        
        // Walk-forward analysis
        std::vector<BacktestResult> runWalkForwardAnalysis(
            const std::string& pair_symbol,
            std::unique_ptr<sep::trading::QuantumPairTrainer> model);
        
        // Monte Carlo simulation
        std::vector<BacktestResult> runMonteCarloSimulation(
            const BacktestResult& baseline_result,
            size_t num_simulations = 1000);
        
        // Stress testing
        std::map<std::string, BacktestResult> runStressTests(
            const std::string& pair_symbol,
            std::unique_ptr<sep::trading::QuantumPairTrainer> model);
        
        // Performance analysis
        void analyzeResults(const BacktestResult& result);
        std::string generateComprehensiveReport(const BacktestResult& result);
        void exportTradeLog(const BacktestResult& result, const std::string& filename);
        void createEquityCurveChart(const BacktestResult& result, const std::string& filename);
        
        // Validation functions
        bool validateAccuracyClaim(const BacktestResult& result, double claimed_accuracy);
        std::string gradeBacktestPerformance(const BacktestResult& result);
        std::vector<std::string> generateRecommendations(const BacktestResult& result);
        
        // Configuration management
        void updateConfig(const BacktestConfig& config);
        BacktestConfig getCurrentConfig() const;
        
    private:
        // Core backtesting implementation
        BacktestResult executeBacktest(
            const std::string& pair_symbol,
            sep::trading::QuantumPairTrainer* model,
            const std::vector<sep::connectors::MarketData>& historical_data);
        
        // Trade execution simulation
        TradeRecord executeTradeSimulation(
            const sep::connectors::MarketData& entry_signal,
            const std::vector<sep::connectors::MarketData>& future_data,
            double confidence_score,
            const std::string& signal_type);
        
        // Performance calculation methods
        void calculateBasicMetrics(BacktestResult& result);
        void calculateRiskMetrics(BacktestResult& result);
        void calculateAdvancedMetrics(BacktestResult& result);
        void calculateTemporalAnalysis(BacktestResult& result);
        void calculateModelAccuracy(BacktestResult& result);
        
        // Risk management simulation
        double calculatePositionSize(
            double account_equity, double confidence_score, double atr);
        bool shouldClosePosition(
            const TradeRecord& open_trade,
            const sep::connectors::MarketData& current_data,
            double current_equity);
        
        // Data preparation and utilities
        std::vector<sep::connectors::MarketData> fetchHistoricalData(
            const std::string& pair_symbol,
            const std::string& start_date,
            const std::string& end_date);
        
        double calculateTradingCosts(
            double entry_price, double exit_price, double position_size,
            const std::string& trade_type);
        
        std::string formatTimestamp(size_t timestamp);
        double calculateATR(const std::vector<sep::connectors::MarketData>& data, size_t periods = 14);
        
        // Reporting utilities
        void generateHTMLReport(const BacktestResult& result, const std::string& filename);
        void generateCSVReport(const BacktestResult& result, const std::string& filename);
        std::string formatCurrency(double amount);
        std::string formatPercentage(double percentage);
        
        // Configuration and state
        BacktestConfig config_;
        std::unique_ptr<sep::connectors::OandaConnector> data_connector_;
        
        // Performance tracking
        mutable std::mutex results_mutex_;
        std::map<std::string, BacktestResult> cached_results_;
        
        // Utilities for statistical calculations
        double calculateSharpeRatio(const std::vector<double>& returns, double risk_free_rate = 0.02);
        double calculateMaxDrawdown(const std::vector<std::pair<std::string, double>>& equity_curve);
        double calculateVolatility(const std::vector<double>& returns);
        double calculateValueAtRisk(const std::vector<double>& returns, double confidence_level = 0.95);
    };

    // Factory function
    std::unique_ptr<BacktestingEngine> createBacktestingEngine(
        const BacktestConfig& config = {});
    
    // Utility functions for backtesting analysis
    std::string generateBacktestSummary(const BacktestResult& result);
    bool compareBacktestResults(const BacktestResult& result1, const BacktestResult& result2);
    BacktestResult aggregateBacktestResults(const std::vector<BacktestResult>& results);

} // namespace sep::backtesting