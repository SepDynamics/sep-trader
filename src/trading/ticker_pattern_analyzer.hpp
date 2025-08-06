#pragma once
#include "engine/internal/standard_includes.h"

#include <string>
#include <vector>
#include <array>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>

#include "quantum/bitspace/qfh.h"
#include "quantum/quantum_manifold_optimizer.h"
#include "connectors/oanda_connector.h"
#include "core_types/result.h"

namespace sep::trading {

/**
 * Advanced Ticker Pattern Analysis Results
 * Contains comprehensive pattern analysis for a single currency pair
 */
struct TickerPatternAnalysis {
    std::string ticker_symbol;
    std::chrono::system_clock::time_point analysis_timestamp;
    
    // Core quantum metrics
    double coherence_score = 0.0;        // Quantum field coherence (0-1)
    double entropy_level = 0.0;          // Information entropy (0-1)  
    double stability_index = 0.0;        // Pattern stability (0-1)
    double rupture_probability = 0.0;    // QFH rupture likelihood (0-1)
    
    // Advanced pattern metrics
    double trend_strength = 0.0;         // Directional trend strength
    double volatility_factor = 0.0;      // Market volatility measure
    double regime_confidence = 0.0;      // Market regime detection confidence
    double correlation_index = 0.0;      // Cross-pair correlation strength
    
    // Multi-timeframe analysis
    struct TimeframeAnalysis {
        std::string timeframe;  // "M1", "M5", "M15", "H1", "H4", "D1"
        double pattern_strength = 0.0;
        double signal_quality = 0.0;
        bool breakout_detected = false;
        bool reversal_pattern = false;
        double confidence_level = 0.0;
    };
    
    std::vector<TimeframeAnalysis> timeframe_results;
    
    // Pattern classification
    enum class PatternType {
        TRENDING_UP,
        TRENDING_DOWN, 
        RANGING,
        BREAKOUT_PENDING,
        REVERSAL_FORMING,
        HIGH_VOLATILITY,
        CONSOLIDATION,
        UNKNOWN
    };
    
    PatternType dominant_pattern = PatternType::UNKNOWN;
    std::vector<PatternType> secondary_patterns;
    
    // Trading signals
    enum class SignalStrength { NONE, WEAK, MODERATE, STRONG, VERY_STRONG };
    enum class SignalDirection { HOLD, BUY, SELL };
    
    SignalDirection primary_signal = SignalDirection::HOLD;
    SignalStrength signal_strength = SignalStrength::NONE;
    double signal_confidence = 0.0;
    
    // Risk assessment
    double estimated_risk_level = 0.0;   // Risk assessment (0-1)
    double maximum_drawdown_risk = 0.0;  // Potential drawdown estimate
    double position_size_recommendation = 0.0;  // Recommended position size
    
    // Pattern evolution tracking
    std::vector<std::string> pattern_evolution_path;  // Historical pattern progression
    std::string predicted_next_pattern;              // AI prediction of next pattern
    double evolution_confidence = 0.0;               // Confidence in evolution prediction
    
    // Metadata
    size_t data_points_analyzed = 0;
    std::chrono::milliseconds analysis_duration{0};
    bool analysis_successful = true;
    std::string error_message;
};

/**
 * Comprehensive Pattern Analysis Configuration
 */
struct PatternAnalysisConfig {
    // Analysis depth settings
    size_t lookback_hours = 24;          // How far back to analyze
    size_t pattern_window_size = 50;     // Size of pattern analysis window
    size_t quantum_analysis_depth = 100; // Depth of quantum field analysis
    
    // Multi-timeframe settings
    std::vector<std::string> timeframes = {"M1", "M5", "M15", "H1"};
    bool enable_cross_timeframe_validation = true;
    double min_cross_timeframe_agreement = 0.60; // 60% agreement threshold
    
    // Signal generation parameters
    double min_signal_confidence = 0.65;   // Minimum confidence for signal generation
    double volatility_adjustment_factor = 1.0;  // Adjust for market volatility
    bool enable_regime_detection = true;
    
    // Risk management
    double max_risk_per_trade = 0.02;      // 2% max risk per trade
    double correlation_risk_limit = 0.8;   // Max correlation with other pairs
    bool enable_drawdown_protection = true;
    
    // Performance optimization
    bool enable_cuda_acceleration = true;
    size_t parallel_processing_threads = 8;
    bool cache_intermediate_results = true;
    
    // Advanced features
    bool enable_sentiment_analysis = false;  // Future feature
    bool enable_news_integration = false;    // Future feature
    bool enable_ai_pattern_prediction = true;
};

/**
 * Advanced Ticker Pattern Analyzer
 * Provides comprehensive quantum-enhanced pattern analysis for currency pairs
 */
class TickerPatternAnalyzer {
public:
    explicit TickerPatternAnalyzer(const PatternAnalysisConfig& config = {});
    ~TickerPatternAnalyzer();

    // Core analysis interface
    TickerPatternAnalysis analyzeTicker(const std::string& ticker_symbol);
    std::future<TickerPatternAnalysis> analyzeTickerAsync(const std::string& ticker_symbol);
    
    // Batch analysis
    std::vector<TickerPatternAnalysis> analyzeMultipleTickers(
        const std::vector<std::string>& ticker_symbols);
    std::future<std::vector<TickerPatternAnalysis>> analyzeMultipleTickersAsync(
        const std::vector<std::string>& ticker_symbols);
    
    // Real-time analysis
    void startRealTimeAnalysis(const std::string& ticker_symbol);
    void stopRealTimeAnalysis(const std::string& ticker_symbol);
    TickerPatternAnalysis getLatestAnalysis(const std::string& ticker_symbol) const;
    
    // Historical analysis
    std::vector<TickerPatternAnalysis> getHistoricalAnalysis(
        const std::string& ticker_symbol,
        std::chrono::system_clock::time_point start_time,
        std::chrono::system_clock::time_point end_time) const;
    
    // Configuration management
    void updateConfig(const PatternAnalysisConfig& config);
    PatternAnalysisConfig getCurrentConfig() const;
    
    // Pattern comparison and correlation
    double calculatePatternSimilarity(const TickerPatternAnalysis& analysis1,
                                     const TickerPatternAnalysis& analysis2);
    std::vector<std::string> findSimilarPatterns(const std::string& ticker_symbol,
                                                double similarity_threshold = 0.8);
    
    // Performance monitoring
    struct AnalysisPerformanceStats {
        std::atomic<size_t> total_analyses{0};
        std::atomic<size_t> successful_analyses{0};
        std::atomic<double> average_analysis_time_ms{0.0};
        std::atomic<size_t> cache_hits{0};
        std::atomic<size_t> cache_misses{0};
        std::chrono::system_clock::time_point last_reset;
        
        // Copy constructor
        AnalysisPerformanceStats(const AnalysisPerformanceStats& other) 
            : total_analyses(other.total_analyses.load()),
              successful_analyses(other.successful_analyses.load()),
              average_analysis_time_ms(other.average_analysis_time_ms.load()),
              cache_hits(other.cache_hits.load()),
              cache_misses(other.cache_misses.load()),
              last_reset(other.last_reset) {}
              
        // Move constructor  
        AnalysisPerformanceStats(AnalysisPerformanceStats&& other) noexcept
            : total_analyses(other.total_analyses.load()),
              successful_analyses(other.successful_analyses.load()),
              average_analysis_time_ms(other.average_analysis_time_ms.load()),
              cache_hits(other.cache_hits.load()),
              cache_misses(other.cache_misses.load()),
              last_reset(std::move(other.last_reset)) {}
              
        // Assignment operator
        AnalysisPerformanceStats& operator=(const AnalysisPerformanceStats& other) {
            if (this != &other) {
                total_analyses.store(other.total_analyses.load());
                successful_analyses.store(other.successful_analyses.load());
                average_analysis_time_ms.store(other.average_analysis_time_ms.load());
                cache_hits.store(other.cache_hits.load());
                cache_misses.store(other.cache_misses.load());
                last_reset = other.last_reset;
            }
            return *this;
        }
        
        // Default constructor
        AnalysisPerformanceStats() = default;
    };
    
    AnalysisPerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();
    
    // Advanced pattern insights
    std::vector<std::string> getPatternInsights(const std::string& ticker_symbol);
    std::map<std::string, double> getPatternCorrelations(const std::string& ticker_symbol);
    
private:
    // Core analysis implementation
    TickerPatternAnalysis performQuantumAnalysis(const std::string& ticker_symbol);
    
    // Data acquisition
    std::vector<sep::connectors::MarketData> fetchMarketData(
        const std::string& ticker_symbol, size_t hours_back);
    
    // Quantum pattern processing
    sep::quantum::QFHResult performQFHAnalysis(
        const std::vector<sep::connectors::MarketData>& market_data);
    std::vector<sep::quantum::Pattern> extractPatterns(
        const std::vector<sep::connectors::MarketData>& market_data);
    
    // Multi-timeframe analysis
    std::vector<TickerPatternAnalysis::TimeframeAnalysis> analyzeTimeframes(
        const std::string& ticker_symbol);
    bool validateCrossTimeframeSignals(
        const std::vector<TickerPatternAnalysis::TimeframeAnalysis>& timeframe_analyses);
    
    // Pattern classification
    TickerPatternAnalysis::PatternType classifyPattern(
        const std::vector<sep::connectors::MarketData>& market_data,
        const sep::quantum::QFHResult& qfh_result);
    
    // Signal generation
    void generateTradingSignals(TickerPatternAnalysis& analysis);
    double calculateSignalConfidence(const TickerPatternAnalysis& analysis);
    
    // Risk assessment
    void performRiskAssessment(TickerPatternAnalysis& analysis);
    double calculateDrawdownRisk(const std::vector<sep::connectors::MarketData>& market_data);
    
    // Pattern evolution prediction
    void predictPatternEvolution(TickerPatternAnalysis& analysis);
    std::string getNextPatternPrediction(const std::vector<std::string>& evolution_path);
    
    // Caching and persistence
    std::string getCacheKey(const std::string& ticker_symbol) const;
    void cacheAnalysisResult(const TickerPatternAnalysis& analysis);
    std::optional<TickerPatternAnalysis> getCachedAnalysis(const std::string& cache_key) const;
    
    // Configuration and state
    PatternAnalysisConfig config_;
    mutable std::mutex config_mutex_;
    
    // Component instances
    std::unique_ptr<sep::quantum::QFHBasedProcessor> qfh_processor_;
    std::unique_ptr<sep::quantum::manifold::QuantumManifoldOptimizer> manifold_optimizer_;
    std::unique_ptr<sep::connectors::OandaConnector> oanda_connector_;
    
    // Real-time analysis state
    std::map<std::string, std::atomic<bool>> real_time_active_;
    std::map<std::string, std::thread> real_time_threads_;
    std::map<std::string, TickerPatternAnalysis> latest_analyses_;
    mutable std::mutex real_time_mutex_;
    
    // Analysis cache
    mutable std::map<std::string, TickerPatternAnalysis> analysis_cache_;
    mutable std::mutex cache_mutex_;
    
    // Performance tracking
    mutable AnalysisPerformanceStats performance_stats_;
    mutable std::mutex stats_mutex_;
    
    // Pattern correlation matrix
    std::map<std::pair<std::string, std::string>, double> pattern_correlations_;
    mutable std::mutex correlations_mutex_;
};

} // namespace sep::trading
