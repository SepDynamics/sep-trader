#pragma once

#include <string>
#include <array>

namespace sep {
namespace training {

// Training mode enumeration
enum class TrainingMode {
    QUICK,          // Fast training for development
    FULL,           // Complete training with optimization
    LIVE_TUNE,      // Live parameter tuning for active trader
    BATCH           // Batch processing for multiple pairs
};

// Pattern quality enumeration
enum class PatternQuality {
    HIGH,           // >70% accuracy, ready for live trading
    MEDIUM,         // 60-70% accuracy, suitable for testing
    LOW,            // <60% accuracy, needs retraining
    UNKNOWN         // Not yet evaluated
};

// Key-value pair structure for metrics and parameters
struct KeyValuePair {
    std::string key;
    std::string value;
    
    KeyValuePair() = default;
    KeyValuePair(const std::string& k, const std::string& v) : key(k), value(v) {}
};

// Training result structure
struct TrainingResult {
    std::string pair;              // Currency pair
    double accuracy;               // Training accuracy
    double stability_score;        // Stability metric
    double coherence_score;        // Coherence metric
    double entropy_score;          // Entropy metric
    std::string model_hash;        // Hash of the trained model
    std::string timestamp;         // Training timestamp (ISO 8601 format)
    PatternQuality quality;        // Quality assessment
    std::array<KeyValuePair, 16> parameters;  // Training parameters
    std::size_t param_count;      // Number of valid parameters

    TrainingResult() 
        : accuracy(0.0)
        , stability_score(0.0)
        , coherence_score(0.0)
        , entropy_score(0.0)
        , quality(PatternQuality::UNKNOWN)
        , param_count(0) {}
};

// Remote trader configuration
struct RemoteTraderConfig {
    std::string host;             // Remote trader host (e.g. Tailscale IP)
    int port;                     // Port number
    bool ssl_enabled;             // Whether to use SSL
    std::string auth_token;       // Authentication token
    int sync_interval_seconds;    // How often to sync patterns (in seconds)
    std::array<KeyValuePair, 16> additional_params;  // Additional parameters
    std::size_t param_count;      // Number of valid parameters

    RemoteTraderConfig() 
        : port(0)
        , ssl_enabled(false)
        , sync_interval_seconds(300)  // Default 5 minutes
        , param_count(0) {}
};

} // namespace training
} // namespace sep