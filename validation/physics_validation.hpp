#pragma once

#include <array>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <limits>

namespace sep::physics_validation {

// Core definitions from the validation plan
constexpr size_t STATE_SIZE = 64;
typedef std::array<bool, STATE_SIZE> BinaryState;

/**
 * @brief Core SEP metrics as defined in physics validation plan
 */
struct SEPMetrics {
    float coherence;  // Baseline-corrected coherence [0,1]
    float stability;  // S_t = 1 - EMA_β[f_t] [0,1] 
    float entropy;    // Bit-level entropy [0,1]
    
    // Intermediate values for analysis
    uint32_t overlap;    // O_t = popcount(s_{t-1} ∧ s_t)
    uint32_t flips;      // F_t = popcount(s_{t-1} ⊕ s_t)
    float flip_rate;     // f_t = F_t/64
    
    SEPMetrics() : coherence(0.0f), stability(0.0f), entropy(0.0f), 
                   overlap(0), flips(0), flip_rate(0.0f) {}
};

/**
 * @brief Triad trajectory point (H, C, S)
 */
struct TriadPoint {
    float entropy;    // H
    float coherence;  // C  
    float stability;  // S
    double timestamp; // Time coordinate
    
    TriadPoint() : entropy(0.0f), coherence(0.0f), stability(0.0f), timestamp(0.0) {}
    TriadPoint(float h, float c, float s, double t) : entropy(h), coherence(c), stability(s), timestamp(t) {}
};

/**
 * @brief Bit mapping strategies from validation plan
 */
enum class BitMappingType {
    D1_DERIVATIVE_SIGN,     // Sign of derivative over 64 staggered micro-windows
    D2_QUANTILE_THRESHOLDS, // 64 quantile thresholds of rolling window
    D3_MULTIBAND_FEATURES   // 8 bands × 8 features (bandpass energy > rolling median)
};

/**
 * @brief Configuration for EMA calculations
 */
struct EMAConfig {
    float half_life; // T_{1/2} for EMA calculation
    float beta;      // β = 1 - exp(-ln(2)/T_{1/2})
    
    EMAConfig(float t_half = 10.0f) : half_life(t_half) {
        beta = 1.0f - std::exp(-std::log(2.0f) / half_life);
    }
};

/**
 * @brief Signal-to-binary converter implementing D1, D2, D3 mappings
 */
class SignalToBinaryConverter {
public:
    explicit SignalToBinaryConverter(BitMappingType type = BitMappingType::D1_DERIVATIVE_SIGN);
    
    BinaryState convert(const float* signal, size_t signal_length, size_t window_size = 64);
    BinaryState convertPoint(float value);
    void reset();
    void setMappingType(BitMappingType type) { mapping_type_ = type; }
    
private:
    BitMappingType mapping_type_;
    float signal_buffer_[1024];
    size_t buffer_size_;
    float quantile_thresholds_[STATE_SIZE];
    
    BinaryState applyD1Mapping(const float* signal, size_t length);
    BinaryState applyD2Mapping(const float* signal, size_t length);
    BinaryState applyD3Mapping(const float* signal, size_t length);
};

/**
 * @brief Core SEP metrics calculator following validation plan definitions
 */
class SEPMetricsCalculator {
public:
    explicit SEPMetricsCalculator(const EMAConfig& config = EMAConfig{});
    
    SEPMetrics calculateMetrics(const BinaryState& prev_state, const BinaryState& curr_state);
    
    static float calculateCoherence(uint32_t overlap, uint32_t n_prev, uint32_t n_curr);
    float updateStability(float flip_rate);
    float calculateEntropy(const BinaryState& state);
    
    void reset();
    void setEMAConfig(const EMAConfig& config);
    
private:
    EMAConfig ema_config_;
    float stability_ema_;
    float bit_probabilities_[STATE_SIZE];
    bool initialized_;
    
    static uint32_t popcount(const BinaryState& state);
    static BinaryState bitwiseAnd(const BinaryState& a, const BinaryState& b);
    static BinaryState bitwiseXor(const BinaryState& a, const BinaryState& b);
};

/**
 * @brief Triad trajectory analyzer for pattern evolution
 */
class TriadAnalyzer {
public:
    static void computeTrajectory(
        const float* signal,
        size_t signal_length,
        SignalToBinaryConverter& converter,
        SEPMetricsCalculator& calculator,
        TriadPoint* trajectory,
        size_t* trajectory_length
    );
    
    static float alignTrajectories(
        const TriadPoint* trajectory1,
        size_t length1,
        const TriadPoint* trajectory2,
        size_t length2,
        float scale_factor = 1.0f
    );
    
    static float calculateRMSE(
        const TriadPoint* traj1,
        size_t length1,
        const TriadPoint* traj2,
        size_t length2
    );
    
    static void timeScale(
        const TriadPoint* input_trajectory,
        size_t input_length,
        float scale_factor,
        TriadPoint* output_trajectory
    );
};

/**
 * @brief Statistical utilities for hypothesis testing
 */
class StatisticalUtils {
public:
    struct BootstrapResult {
        float lower_bound;
        float median;
        float upper_bound;
    };
    
    static BootstrapResult bootstrapCI(
        const float* data,
        size_t data_length,
        size_t n_bootstrap = 10000,
        float confidence = 0.95f
    );
    
    static float wilcoxonTest(
        const float* group1,
        const float* group2,
        size_t group_size
    );
    
    static float dtwDistance(
        const TriadPoint* traj1,
        size_t length1,
        const TriadPoint* traj2,
        size_t length2
    );
    
    static void benjaminiHochbergCorrection(
        const float* p_values,
        size_t num_p_values,
        bool* significant,
        float q_value = 0.05f
    );
};

} // namespace sep::physics_validation